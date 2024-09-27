use anyhow::{anyhow, Context, Result};
use aws_sdk_s3::presigning::PresigningConfig;
use faer::Col;
use log::info;
use std::fs::File;
use std::io::{BufReader, Seek, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tokio::time::Duration;
use tokio::time::Instant;
use zip::write::SimpleFileOptions;
use zip::CompressionMethod;

use crate::igwas::{run_igwas_df_impl, Projection};
use crate::models::RequestMetadata;
use crate::phenotype_definitions::format_phenotype_definition;
use crate::regression::regress_left_inverse_vec;
use crate::AppState;
use crate::{
    models::{WebGWASRequestId, WebGWASResultStatus},
    phenotype_definitions::apply_phenotype_definition,
};

pub fn worker_loop(state: Arc<AppState>) {
    loop {
        let task = {
            let mut queue = state.queue.lock().unwrap();
            queue.pop()
        };
        if let Some(request) = task {
            info!("Got a request");
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
    // 1. Apply the phenotype and compute the projection coefficents
    let cohort_info = {
        let binding = state.cohort_id_to_data.lock().unwrap();
        binding.get(&request.cohort_id).unwrap().clone()
    };
    let phenotype =
        apply_phenotype_definition(&request.phenotype_definition, &cohort_info.features_df)
            .context(anyhow!("Failed to apply phenotype definition"))?;
    let mut phenotype_mat = Col::zeros(phenotype.len());
    phenotype.f32()?.iter().enumerate().for_each(|(i, x)| {
        phenotype_mat[i] = x.expect("Failed to get phenotype value");
    });
    let start = Instant::now();
    let mut beta = regress_left_inverse_vec(&phenotype_mat, &cohort_info.left_inverse);
    // Drop the last element of the beta vector, which is the intercept
    beta.truncate(beta.nrows() - 1);
    let duration = start.elapsed();
    info!("Regression took {:?}", duration);
    let phenotype_names: Vec<String> = cohort_info
        .features_df
        .drop("intercept")?
        .get_column_names()
        .iter()
        .map(|x| x.to_string())
        .collect();
    let mut projection = Projection::new(phenotype_names, beta.clone())?;

    // 2. Compute the projection variance
    let start = Instant::now();
    let projection_variance = beta.transpose() * &cohort_info.covariance_matrix * &beta;
    let duration = start.elapsed();
    info!("Computing projection variance took {:?}", duration);

    // 3. Compute GWAS
    let output_path = state
        .root_directory
        .join(format!("results/{}.tsv.zst", request.id));
    let start = Instant::now();
    run_igwas_df_impl(
        &cohort_info.gwas_df,
        &mut projection,
        projection_variance,
        cohort_info.cohort.num_covar.expect("Num_covar is missing") as usize,
        &output_path,
        16,
    )?;
    let duration = start.elapsed();
    info!("GWAS took {:?}", duration);
    let total_duration = request.request_time.elapsed();
    {
        let mut results = state.results.lock().unwrap();
        let result = results.get_mut(&request.id).unwrap();
        result.status = WebGWASResultStatus::Uploading;
        result.local_result_file = Some(output_path.clone());
    }

    info!("Writing metadata");
    let metadata_file = create_metadata_file(&state, &request)?;
    info!("Wrote metadata");
    info!("Creating zip file");
    let output_zip_path = create_output_zip(&output_path, &metadata_file)?;
    info!("Created zip file");
    std::fs::remove_file(metadata_file)?;

    let url = if state.settings.dry_run {
        info!("Dry run, skipping S3 upload");
        None
    } else {
        let start = Instant::now();
        let key = format!("{}/{}.zip", state.settings.s3_result_path, request.id);
        let rt = tokio::runtime::Runtime::new()?;
        let url = rt.block_on(async {
            upload_object(
                &state.s3_client,
                &output_zip_path,
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
        std::fs::remove_file(output_zip_path)?;
        Some(url)
    };

    info!("Overall took {:?}", total_duration);
    {
        let mut results = state.results.lock().unwrap();
        let result = results.get_mut(&request.id).unwrap();
        result.status = WebGWASResultStatus::Done;
        result.url = url;
    }
    Ok(())
}

pub async fn upload_object(
    client: &aws_sdk_s3::Client,
    file_name: &Path,
    bucket_name: &str,
    key: &str,
) -> Result<aws_sdk_s3::operation::put_object::PutObjectOutput> {
    let body = aws_sdk_s3::primitives::ByteStream::from_path(file_name).await?;
    let result = client
        .put_object()
        .bucket(bucket_name)
        .key(key)
        .body(body)
        .send()
        .await?;
    Ok(result)
}

pub fn create_metadata_file(state: &AppState, request: &WebGWASRequestId) -> Result<PathBuf> {
    let cohort_info = {
        let binding = state.cohort_id_to_data.lock().unwrap();
        binding.get(&request.cohort_id).unwrap().clone()
    };
    let metadata = RequestMetadata::new(
        request.id,
        format_phenotype_definition(&request.phenotype_definition),
        cohort_info.cohort.name.clone(),
        cohort_info.features_df.height(),
    );
    let output_metadata_path = state
        .root_directory
        .join(format!("results/{}.txt", request.id));
    let mut metadata_file = File::create(output_metadata_path.clone())?;
    write!(metadata_file, "{}", metadata)?;
    Ok(output_metadata_path)
}

pub fn add_file_to_zip<W>(
    zip_writer: &mut zip::ZipWriter<W>,
    file_path: &Path,
    name_in_zip: &str,
) -> zip::result::ZipResult<()>
where
    W: Write + Seek,
{
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o644);
    let file = File::open(file_path)?;
    let mut buffered_reader = BufReader::new(file);
    zip_writer.start_file(name_in_zip, options)?;
    std::io::copy(&mut buffered_reader, zip_writer)?;
    Ok(())
}

pub fn create_output_zip(output_path: &Path, metadata_path: &Path) -> Result<PathBuf> {
    let output_zip_path = output_path.with_extension("").with_extension("zip");
    let mut zip_writer = zip::ZipWriter::new(File::create(output_zip_path.clone())?);
    add_file_to_zip(&mut zip_writer, output_path, "results.tsv.zst")?;
    add_file_to_zip(&mut zip_writer, metadata_path, "metadata.txt")?;
    zip_writer.finish()?;
    Ok(output_zip_path)
}
