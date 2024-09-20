use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{Path, Query, State},
    routing::{get, post, put},
    Json, Router,
};
use faer::Col;
use faer_ext::polars::polars_to_faer_f32;
use itertools::izip;
use log::info;
use phenotype_definitions::{apply_phenotype_definition, validate_phenotype_definition};
use polars::prelude::*;
use std::{iter::repeat, sync::Arc};
use std::{path::PathBuf, thread};
use tokio::time::Instant;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use webgwas_backend::models::{
    ApproximatePhenotypeValues, CohortResponse, FeatureResponse, GetFeaturesRequest,
    PhenotypeFitQuality, PhenotypeSummary, PvaluesResponse, ValidPhenotypeResponse, WebGWASRequest,
    WebGWASRequestId, WebGWASResponse, WebGWASResult, WebGWASResultStatus,
};
use webgwas_backend::phenotype_definitions;
use webgwas_backend::regression::regress_left_inverse_vec;
use webgwas_backend::AppState;
use webgwas_backend::{config::Settings, models::PhenotypeSummaryRequest};
use webgwas_backend::{errors::WebGWASError, worker::worker_loop};

#[tokio::main]
async fn main() {
    env_logger::init();

    info!("Reading settings");
    let settings = Settings::read_file("settings.toml")
        .context("Failed to read config file")
        .unwrap();
    info!("Initializing state");
    let state = Arc::new(AppState::new(settings).await.unwrap());

    let worker_state = state.clone();
    thread::spawn(move || {
        worker_loop(worker_state);
    });

    info!("Initializing app");
    let app = Router::new()
        .route("/api/cohorts", get(get_cohorts))
        .route("/api/features", get(get_features))
        .route("/api/phenotype", put(validate_phenotype))
        .route("/api/phenotype_summary", post(get_phenotype_summary))
        .route("/api/igwas", post(post_igwas))
        .route("/api/igwas/results/:request_id", get(get_igwas_results))
        .route(
            "/api/igwas/results/pvalues/:request_id",
            get(get_igwas_pvalues),
        )
        .layer(CorsLayer::permissive())
        .with_state(state);

    info!("Starting server");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Get all cohorts
async fn get_cohorts(State(state): State<Arc<AppState>>) -> Json<Vec<CohortResponse>> {
    info!("Fetching cohorts");
    let result = sqlx::query_as::<_, CohortResponse>("SELECT id, name FROM cohort")
        .fetch_all(&state.db)
        .await
        .context("Failed to fetch cohorts")
        .unwrap();
    Json(result)
}

/// Get all features for a given cohort
async fn get_features(
    request: Query<GetFeaturesRequest>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<FeatureResponse>>, WebGWASError> {
    let request = request.0;
    info!("Fetching features for cohort {}", request.cohort_id);
    let result = sqlx::query_as::<_, FeatureResponse>(
        "SELECT code, name, type as node_type, sample_size
        FROM feature WHERE cohort_id = $1
        ORDER BY sample_size DESC",
    )
    .bind(request.cohort_id)
    .fetch_all(&state.db)
    .await
    .context("Failed to fetch features")?;

    Ok(Json(result))
}

/// Validate a phenotype definition
async fn validate_phenotype(
    State(state): State<Arc<AppState>>,
    Json(request): Json<WebGWASRequest>,
) -> Json<ValidPhenotypeResponse> {
    info!("Validating phenotype definition");
    let start = Instant::now();
    let result = match validate_phenotype_definition(
        request.cohort_id,
        &request.phenotype_definition,
        &state.knowledge_base,
    ) {
        Ok(_) => Json(ValidPhenotypeResponse {
            is_valid: true,
            message: "Phenotype definition is valid".to_string(),
            phenotype_definition: request.phenotype_definition,
        }),
        Err(err) => Json(ValidPhenotypeResponse {
            is_valid: false,
            message: format!("Phenotype definition is invalid: {}", err),
            phenotype_definition: request.phenotype_definition,
        }),
    };
    let duration = start.elapsed();
    info!("Validating phenotype definition took {:?}", duration);
    result
}

async fn get_phenotype_summary(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PhenotypeSummaryRequest>,
) -> Result<Json<PhenotypeSummary>, WebGWASError> {
    let overall_start = Instant::now();
    // 1. Validate the phenotype definition
    let definition = match validate_phenotype_definition(
        request.cohort_id,
        &request.phenotype_definition,
        &state.knowledge_base,
    ) {
        Ok(definition) => definition,
        Err(err) => {
            return Err(anyhow!("Failed to validate phenotype definition: {}", err).into());
        }
    };
    // 2. Apply the phenotype definition
    let cohort_info = {
        let binding = state.cohort_id_to_data.lock().unwrap();
        binding.get(&request.cohort_id).unwrap().clone()
    };
    let phenotype = apply_phenotype_definition(&definition, &cohort_info.features_df)
        .context(anyhow!("Failed to apply phenotype definition"))?;
    let mut phenotype_mat = Col::zeros(phenotype.len());
    phenotype.f32()?.iter().enumerate().for_each(|(i, x)| {
        phenotype_mat[i] = x.expect("Failed to get phenotype value");
    });

    // 3. Regress the phenotype against the features
    let start = Instant::now();
    let beta = regress_left_inverse_vec(&phenotype_mat, &cohort_info.left_inverse);
    let duration = start.elapsed();
    info!("Regression took {:?}", duration);

    let start = Instant::now();
    let phenotype_pred = polars_to_faer_f32(cohort_info.features_df.clone().lazy())? * &beta;
    let duration = start.elapsed();
    info!("Prediction took {:?}", duration);

    let phenotype_pred_df = DataFrame::new(vec![
        Column::new(
            "approx_value".into(),
            phenotype_pred.iter().collect::<Series>(),
        ),
        Column::new("true_value".into(), phenotype),
        Column::new(
            "count".into(),
            repeat(1).take(phenotype_pred.nrows()).collect::<Series>(),
        ),
    ])?
    .group_by(["approx_value", "true_value"])?
    .select(["count"])
    .count()?;
    let phenotype_values = izip!(
        phenotype_pred_df.column("true_value")?.f32()?.into_iter(),
        phenotype_pred_df.column("approx_value")?.f32()?.into_iter(),
        phenotype_pred_df.column("count_count")?.u32()?.into_iter(),
    )
    .map(|(x, y, n)| {
        Some(ApproximatePhenotypeValues {
            true_value: x?,
            approx_value: y?,
            n: n? as i32,
        })
    })
    .take(request.n_samples.unwrap_or(phenotype_pred.nrows()))
    .collect::<Option<Vec<ApproximatePhenotypeValues>>>()
    .context("Failed to calculate phenotype values")?;

    // 4. Calculate the fit quality
    let start = Instant::now();
    let rss = (&phenotype_pred - &phenotype_mat)
        .iter()
        .map(|x| x.powi(2))
        .sum::<f32>();
    let mean = phenotype_mat.sum() / phenotype_mat.nrows() as f32;
    let tss = phenotype_mat
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>();
    let r2 = 1.0 - (rss / tss);
    let duration = start.elapsed();
    info!("Fit quality took {:?}", duration);
    let fit_quality_reference = state
        .fit_quality_reference
        .iter()
        .take(request.n_samples.unwrap_or(phenotype_pred.nrows()))
        .cloned()
        .collect::<Vec<PhenotypeFitQuality>>();

    let total_duration = overall_start.elapsed();
    info!("Overall took {:?}", total_duration);

    Ok(Json(PhenotypeSummary {
        phenotype_definition: request.phenotype_definition,
        cohort_id: request.cohort_id,
        phenotype_values,
        fit_quality_reference,
        rsquared: r2,
    }))
}

async fn post_igwas(
    State(state): State<Arc<AppState>>,
    Json(request): Json<WebGWASRequest>,
) -> Json<WebGWASResponse> {
    let request_time = Instant::now();
    info!("Received webgwas request");
    // Validate the cohort and phenotype
    let unique_id = Uuid::new_v4();
    match validate_phenotype_definition(
        request.cohort_id,
        &request.phenotype_definition,
        &state.knowledge_base,
    ) {
        Ok(definition) => {
            // Build the processed request
            let request = WebGWASRequestId {
                id: unique_id,
                request_time,
                phenotype_definition: definition,
                cohort_id: request.cohort_id,
            };
            // Put the request in the queue
            state.queue.lock().unwrap().push(request);
            // Return the request id
            Json(WebGWASResponse {
                request_id: unique_id,
                status: WebGWASResultStatus::Queued,
                message: None,
            })
        }
        Err(err) => Json(WebGWASResponse {
            request_id: unique_id,
            status: WebGWASResultStatus::Error,
            message: Some(format!("Failed to validate phenotype definition: {}", err)),
        }),
    }
}

async fn get_igwas_results(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<Uuid>,
) -> Json<WebGWASResult> {
    info!("Fetching WebGWAS result for {}", request_id);
    match state.results.lock().unwrap().get(&request_id) {
        Some(result) => Json(result.clone()),
        None => Json(WebGWASResult {
            request_id,
            status: WebGWASResultStatus::Error,
            error_msg: Some(format!("No result found for request {}", request_id)),
            url: None,
            local_result_file: None,
        }),
    }
}

#[axum::debug_handler]
async fn get_igwas_pvalues(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<Uuid>,
) -> Json<PvaluesResponse> {
    info!("Fetching WebGWAS p-values for {}", request_id);
    match state.results.lock().unwrap().get(&request_id) {
        Some(results) => match &results.local_result_file {
            Some(path) => match load_pvalues(path.to_path_buf()) {
                Ok(pvalues) => Json(PvaluesResponse {
                    request_id,
                    status: WebGWASResultStatus::Done,
                    error_msg: None,
                    pvalues: Some(pvalues),
                }),
                Err(err) => Json(PvaluesResponse {
                    request_id,
                    status: WebGWASResultStatus::Error,
                    error_msg: Some(format!("Failed to load p-values: {}", err)),
                    pvalues: None,
                }),
            },
            None => Json(PvaluesResponse {
                request_id,
                status: WebGWASResultStatus::Error,
                error_msg: Some(format!("No local file found for request {}", request_id)),
                pvalues: None,
            }),
        },
        None => Json(PvaluesResponse {
            request_id,
            status: WebGWASResultStatus::Error,
            error_msg: Some(format!("No result found for request {}", request_id)),
            pvalues: None,
        }),
    }
}

fn load_pvalues(path: PathBuf) -> Result<Vec<f32>> {
    let schema_override = Schema::from_iter(vec![("neg_log_p_value".into(), DataType::Float32)]);
    CsvReadOptions::default()
        .with_schema_overwrite(Some(Arc::new(schema_override)))
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .try_into_reader_with_file_path(Some(path))?
        .finish()?
        .column("neg_log_p_value")?
        .f32()?
        .iter()
        .map(|x| x.context("Failed to get p-value"))
        .collect::<Result<Vec<f32>>>()
}
