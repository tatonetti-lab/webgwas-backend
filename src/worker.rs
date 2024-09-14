use anyhow::{anyhow, Context, Result};
use aws_sdk_s3::presigning::PresigningConfig;
use log::info;
use ndarray::s;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use tokio::time::Duration;
use tokio::time::Instant;
use webgwas::igwas_prod::{run_igwas_df_impl, Projection};

use crate::regression::regress;
use crate::AppState;
use crate::{
    models::{WebGWASRequestId, WebGWASResult, WebGWASResultStatus},
    phenotype_definitions::apply_phenotype_definition,
};

pub fn worker_loop(state: Arc<AppState>) {
    loop {
        let task = {
            let mut queue = state.queue.lock().unwrap();
            queue.pop()
        };
        if let Some(request) = task {
            info!("Got a request!!");
            let result = handle_webgwas_request(state.clone(), request);
            if let Err(err) = result {
                info!("Failed to handle request: {}", err);
            }
        } else {
            thread::sleep(Duration::from_millis(100));
        }
    }
}

pub fn handle_webgwas_request(state: Arc<AppState>, request: WebGWASRequestId) -> Result<()> {
    // Inform the results that the request is queued
    let result = WebGWASResult {
        request_id: request.id,
        status: WebGWASResultStatus::Queued,
        error_msg: None,
        url: None,
    };
    state.results.lock().unwrap().insert(request.id, result);
    // 1. Apply the phenotype and compute the projection coefficents
    let cohort_info = {
        let mut binding = state.cohort_id_to_info.lock().unwrap();
        binding.get(&request.cohort_id).unwrap().clone()
    };
    let phenotype =
        apply_phenotype_definition(&request.phenotype_definition, &cohort_info.features_df)
            .context(anyhow!("Failed to apply phenotype definition"))?;
    let phenotype_ndarray = phenotype
        .f32()
        .context("Failed to convert phenotype to ndarray")?
        .to_ndarray()?
        .into_owned();
    let start = Instant::now();
    let beta = regress(&phenotype_ndarray, &cohort_info.left_inverse);
    // Drop the last element of the beta vector, which is the intercept
    let beta = beta.slice(s![..-1]);
    let duration = start.elapsed();
    info!("Regression took {:?}", duration);
    let phenotype_names: Vec<String> = cohort_info
        .features_df
        .drop("const")?
        .get_column_names()
        .iter()
        .map(|x| x.to_string())
        .collect();
    let mut projection = Projection::new(phenotype_names, beta.to_vec())?;

    // 2. Compute the projection variance
    let start = Instant::now();
    let projection_variance = beta.dot(&cohort_info.covariance_matrix.dot(&beta));
    let duration = start.elapsed();
    info!("Computing projection variance took {:?}", duration);

    // 3. Compute GWAS
    let output_path = format!("{}.tsv.zst", request.id);
    let start = Instant::now();
    run_igwas_df_impl(
        &cohort_info.gwas_df,
        &mut projection,
        projection_variance,
        cohort_info.num_covar as usize,
        output_path.to_string(),
        16,
    )?;
    let duration = start.elapsed();
    info!("GWAS took {:?}", duration);
    let total_duration = request.request_time.elapsed();

    let start = Instant::now();
    let key = format!("{}/{}.tsv.zst", state.settings.s3_result_path, request.id);
    let rt = tokio::runtime::Runtime::new()?;
    let url = rt.block_on(async {
        upload_object(
            &state.s3_client,
            &output_path,
            &state.settings.s3_bucket,
            &key,
        )
        .await
        .context("Failed to upload object")
        .unwrap();
        const URL_EXPIRES_IN: Duration = Duration::from_secs(3600);
        let url = state
            .s3_client
            .get_object()
            .bucket(&state.settings.s3_bucket)
            .key(&key)
            .presigned(PresigningConfig::expires_in(URL_EXPIRES_IN).unwrap())
            .await
            .context("Failed to get presigned URL")
            .unwrap()
            .uri()
            .to_string();
        url
    });
    let duration = start.elapsed();
    info!("Uploading took {:?}", duration);
    // Remove the file
    std::fs::remove_file(output_path).context("Failed to remove the local result file")?;

    info!("Overall took {:?}", total_duration);
    let result = WebGWASResult {
        request_id: request.id,
        status: WebGWASResultStatus::Done,
        error_msg: None,
        url: Some(url),
    };
    state.results.lock().unwrap().insert(request.id, result);
    Ok(())
}

pub async fn upload_object(
    client: &aws_sdk_s3::Client,
    file_name: &str,
    bucket_name: &str,
    key: &str,
) -> Result<aws_sdk_s3::operation::put_object::PutObjectOutput> {
    let body = aws_sdk_s3::primitives::ByteStream::from_path(Path::new(file_name)).await?;
    let result = client
        .put_object()
        .bucket(bucket_name)
        .key(key)
        .body(body)
        .send()
        .await?;
    Ok(result)
}
