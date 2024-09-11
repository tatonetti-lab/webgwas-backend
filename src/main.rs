use anyhow::{anyhow, Context, Result};
use axum::{
    debug_handler,
    extract::{Path, Query, State},
    routing::{get, post, put},
    Json, Router,
};
use itertools::izip;
use log::debug;
use phenotype_definitions::{apply_phenotype_definition, validate_phenotype_definition};
use polars::{
    datatypes::Float32Type,
    frame::DataFrame,
    prelude::{IndexOrder, NamedFrom},
    series::Series,
};
use std::thread;
use std::{iter::repeat, sync::Arc};
use tokio::time::Instant;
use tower_http::cors::CorsLayer;
use uuid::Uuid;
use webgwas_backend_rs::config::Settings;
use webgwas_backend_rs::models::{
    ApproximatePhenotypeValues, Cohort, Feature, GetFeaturesRequest, PhenotypeSummary,
    ValidPhenotypeResponse, WebGWASRequest, WebGWASRequestId, WebGWASResponse, WebGWASResult,
    WebGWASResultStatus,
};
use webgwas_backend_rs::phenotype_definitions;
use webgwas_backend_rs::regression::regress;
use webgwas_backend_rs::AppState;
use webgwas_backend_rs::{config::Settings, models::PhenotypeSummaryRequest};
use webgwas_backend_rs::{errors::WebGWASError, worker::worker_loop};

#[tokio::main]
async fn main() {
    env_logger::init();

    debug!("Reading settings");
    let settings = Settings::read_file("settings.toml")
        .context("Failed to read config file")
        .unwrap();
    debug!("Initializing state");
    let state = Arc::new(AppState::new(settings).await.unwrap());

    let worker_state = state.clone();
    thread::spawn(move || {
        worker_loop(worker_state);
    });

    debug!("Initializing app");
    let app = Router::new()
        .route("/", get(|| async { "Hello, world!" }))
        .route("/api/cohorts", get(get_cohorts))
        .route("/api/features", get(get_features))
        .route("/api/phenotype", put(validate_phenotype))
        .route("/api/phenotype_summary", get(get_phenotype_summary))
        .route("/api/igwas", post(post_igwas))
        .route("/api/igwas/results/:request_id", get(get_igwas_results))
        .layer(CorsLayer::permissive())
        .with_state(state);

    debug!("Starting server");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Get all cohorts
#[debug_handler]
async fn get_cohorts(State(state): State<Arc<AppState>>) -> Json<Vec<Cohort>> {
    let result = sqlx::query_as::<_, Cohort>("SELECT * FROM cohort")
        .fetch_all(&state.db)
        .await
        .context("Failed to fetch cohorts")
        .unwrap();
    Json(result)
}

/// Get all features for a given cohort
#[debug_handler]
async fn get_features(
    request: Query<GetFeaturesRequest>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Feature>>, WebGWASError> {
    let result = sqlx::query_as::<_, Feature>(
        "SELECT id, code, name, type as node_type FROM feature WHERE cohort_id = ?",
    )
    .bind(request.cohort_id)
    .fetch_all(&state.db)
    .await
    .context("Failed to fetch features")?;

    Ok(Json(result))
}

/// Validate a phenotype definition
#[debug_handler]
async fn validate_phenotype(
    State(state): State<Arc<AppState>>,
    Json(request): Json<WebGWASRequest>,
) -> Json<ValidPhenotypeResponse> {
    match validate_phenotype_definition(
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
    }
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
    let cohort_info = state
        .cohort_id_to_info
        .get(&request.cohort_id)
        .unwrap()
        .clone();
    let phenotype = apply_phenotype_definition(&definition, &cohort_info.features_df)
        .context(anyhow!("Failed to apply phenotype definition"))?;
    let phenotype_ndarray = phenotype
        .f32()
        .context("Failed to convert phenotype to ndarray")?
        .to_ndarray()?
        .into_owned();

    // 3. Regress the phenotype against the features
    let start = Instant::now();
    let beta = regress(&phenotype_ndarray, &cohort_info.left_inverse);
    let duration = start.elapsed();
    debug!("Regression took {:?}", duration);

    let start = Instant::now();
    let phenotype_pred = cohort_info
        .features_df
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)?
        .dot(&beta);
    let duration = start.elapsed();
    debug!("Prediction took {:?}", duration);

    let phenotype_pred_df = DataFrame::new(vec![
        Series::new(
            "approx_value".into(),
            phenotype_pred.iter().collect::<Series>(),
        ),
        Series::new("true_value".into(), phenotype),
        Series::new(
            "count".into(),
            repeat(1).take(phenotype_pred.len()).collect::<Series>(),
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
    .take(request.n_samples.unwrap_or(phenotype_pred.len()))
    .collect::<Option<Vec<ApproximatePhenotypeValues>>>()
    .context("Failed to calculate phenotype values")?;

    // 4. Calculate the fit quality
    let start = Instant::now();
    let rss = (&phenotype_pred - &phenotype_ndarray).map(|x| x * x).sum();
    let mean = phenotype_pred.mean().context("Failed to calculate mean")?;
    let tss = (phenotype_ndarray - mean).map(|x| x * x).sum();
    let r2 = rss / tss;
    let duration = start.elapsed();
    debug!("Fit quality took {:?}", duration);
    let fit_quality_reference = state
        .fit_quality_reference
        .iter()
        .take(request.n_samples.unwrap_or(phenotype_pred.len()))
        .cloned()
        .collect::<Vec<PhenotypeFitQuality>>();

    let total_duration = overall_start.elapsed();
    debug!("Overall took {:?}", total_duration);

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
    debug!("Received webgwas request");
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
    debug!("Fetching WebGWAS result for {}", request_id);
    match state.results.lock().unwrap().get(&request_id) {
        Some(result) => Json(result.clone()),
        None => Json(WebGWASResult {
            request_id,
            status: WebGWASResultStatus::Error,
            error_msg: Some(format!("No result found for request {}", request_id)),
            url: None,
        }),
    }
}
