use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{ConnectInfo, Path, Query, State},
    routing::{get, post, put},
    Json, Router,
};
use itertools::izip;
use log::{error, info};
use phenotype_definitions::{apply_phenotype_definition, validate_phenotype_definition};
use sqlx::SqlitePool;
use std::sync::Arc;
use std::{net::SocketAddr, thread};
use tower_http::{compression::CompressionLayer, cors::CorsLayer, trace::TraceLayer};
use tracing::info_span;
use tracing_appender::rolling;
use tracing_subscriber::Layer;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

use webgwas_backend::{models::CacheRow, remove_cache_entry, AppState};
use webgwas_backend::{config::Settings, models::PhenotypeSummaryRequest};
use webgwas_backend::{errors::WebGWASError, worker::worker_loop};
use webgwas_backend::phenotype_definitions;
use webgwas_backend::models::{
        ApproximatePhenotypeValues, CohortResponse, FeatureResponse, GetFeaturesRequest,
        PhenotypeFitQuality, PhenotypeSummary, ValidPhenotypeResponse,
        WebGWASRequest, WebGWASRequestId, WebGWASResponse, WebGWASResult, WebGWASResultStatus,
};
use webgwas_backend::{regression::regress_left_inverse_vec, utils::vec_to_col};

#[tokio::main]
async fn main() {
    let settings = Settings::read_file("settings.toml")
        .context("Failed to read config file")
        .unwrap();

    // Configure tracing/logging
    let appender = rolling::daily(&settings.log_path, "webgwas");
    let subscriber = tracing_subscriber::registry().with(
        fmt::layer()
            .json()
            .flatten_event(true)
            .with_writer(appender)
            .with_thread_ids(false)
            .with_target(false)
            .with_level(true)
            .with_timer(fmt::time::ChronoUtc::rfc_3339())
            .with_span_events(fmt::format::FmtSpan::CLOSE)
            .with_filter(tracing_subscriber::EnvFilter::from_default_env()),
    );
    subscriber.init();

    // Trace layer for the http server
    let trace_layer = TraceLayer::new_for_http().make_span_with(|request: &http::Request<_>| {
        let ip = get_client_ip(request);
        tracing::info_span!(
            "API request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            client_ip = %ip,
        )
    });

    let state = {
        let _span = tracing::info_span!("Initializing state").entered();
        Arc::new(AppState::new(settings).await.unwrap())
    };

    let worker_state = state.clone();
    thread::spawn(move || {
        worker_loop(worker_state);
    });

    let app = Router::new()
        .route("/api/cohorts", get(get_cohorts))
        .route("/api/features", get(get_features))
        .route("/api/phenotype", put(validate_phenotype))
        .route("/api/phenotype_summary", post(get_phenotype_summary))
        .route("/api/igwas", post(post_igwas))
        .route("/api/igwas/results/:request_id", get(get_igwas_results))
        .route(
            "/api/igwas/results/pvalues/:request_id",
            post(webgwas_backend::endpoints::pvalues::get_igwas_pvalues),
        )
        .layer(trace_layer)
        .layer(CorsLayer::permissive())
        .layer(CompressionLayer::new().gzip(true).deflate(true).br(true).zstd(true))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}

/// Get all cohorts
async fn get_cohorts(State(state): State<Arc<AppState>>) -> Json<Vec<CohortResponse>> {
    let result = sqlx::query_as::<_, CohortResponse>("SELECT id, name FROM cohort")
        .fetch_all(&state.db)
        .await
        .context("Failed to fetch cohorts")
        .unwrap();
    Json(result)
}

/// Get all features for a given cohort
async fn get_features(
    Query(request): Query<GetFeaturesRequest>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<FeatureResponse>>, WebGWASError> {
    let result = sqlx::query_as::<_, FeatureResponse>(
        "SELECT code, name, type as node_type, sample_size
        FROM feature WHERE cohort_id = $1
        ORDER BY sample_size DESC",
    )
    .bind(request.cohort_id)
    .fetch_all(&state.db)
    .await;

    match result {
        Ok(result) => Ok(Json(result)),
        Err(err) => {
            error!("Failed to fetch features: {}", err);
            Err(err.into())
        }
    }
}

/// Validate a phenotype definition
async fn validate_phenotype(
    State(state): State<Arc<AppState>>,
    Json(request): Json<WebGWASRequest>,
) -> Json<ValidPhenotypeResponse> {
    let result = match validate_phenotype_definition(
        request.cohort_id,
        &request.phenotype_definition,
        &state.knowledge_base,
    ) {
        Ok(_) => ValidPhenotypeResponse {
            is_valid: true,
            message: "Phenotype definition is valid".to_string(),
            phenotype_definition: request.phenotype_definition,
        },
        Err(err) => ValidPhenotypeResponse {
            is_valid: false,
            message: format!("Phenotype definition is invalid: {}", err),
            phenotype_definition: request.phenotype_definition,
        },
    };
    Json(result)
}

async fn get_phenotype_summary(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PhenotypeSummaryRequest>,
) -> Result<Json<PhenotypeSummary>, WebGWASError> {
    // TODO: Figure out how to reduce memory usage here
    // TODO: Reduce the amount of code duplication here
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
    let phenotype = apply_phenotype_definition(
        &definition,
        &cohort_info.feature_names,
        &cohort_info.features,
    )
    .context(anyhow!("Failed to apply phenotype definition"))?;
    let phenotype_col = vec_to_col(&phenotype);

    // 3. Regress the phenotype against the features
    let (beta, intercept) = {
        let _span = info_span!("regress_left_inverse_vec").entered();
        let mut beta = regress_left_inverse_vec(&phenotype_col, &cohort_info.left_inverse);
        let intercept = beta.read(beta.nrows() - 1);
        beta.truncate(beta.nrows() - 1);
        (beta, intercept)
    };
    let phenotype_pred = {
        let _span = info_span!("polars_to_faer_f32").entered();
        let intercept_col = vec_to_col(&vec![intercept; phenotype_col.nrows()]);
        cohort_info.features.as_ref() * beta + intercept_col
    };

    let phenotype_values = izip!(phenotype.iter(), phenotype_pred.iter())
        .map(|(x, y)| {
            Some(ApproximatePhenotypeValues {
                true_value: *x,
                approx_value: *y,
            })
        })
        .take(request.n_samples.unwrap_or(phenotype_pred.nrows()))
        .collect::<Option<Vec<ApproximatePhenotypeValues>>>()
        .context("Failed to calculate phenotype values")?;

    // 4. Calculate the fit quality
    let rsquared = {
        let _span = info_span!("compute_rsquared").entered();
        let rss = (&phenotype_pred - &phenotype_col)
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>();
        let mean = phenotype_col.sum() / phenotype_col.nrows() as f32;
        let tss = phenotype_col
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>();

        1.0 - (rss / tss) // This will be NaN if tss is zero
    };
    let fit_quality_reference = state
        .fit_quality_reference
        .iter()
        .take(request.n_samples.unwrap_or(phenotype_pred.nrows()))
        .cloned()
        .collect::<Vec<PhenotypeFitQuality>>();

    Ok(Json(PhenotypeSummary {
        phenotype_definition: request.phenotype_definition,
        cohort_id: request.cohort_id,
        phenotype_values,
        fit_quality_reference,
        rsquared,
    }))
}

#[axum::debug_handler]
async fn post_igwas(
    State(state): State<Arc<AppState>>,
    Json(request): Json<WebGWASRequest>,
) -> Json<WebGWASResponse> {
    let unique_id = Uuid::new_v4();
    tracing::info!(
        request_id = %unique_id, cohort_id = %request.cohort_id, 
        phenotype = %request.phenotype_definition, "Received webgwas request");

    let cached_result = get_cached_results(&state.db, request.cohort_id, &request.phenotype_definition).await;
    if let Ok(Some(result)) = cached_result {
        if let Ok(request_id) = Uuid::parse_str(&result.request_id) {
            info!("Found cached result");
            return Json(WebGWASResponse {
                request_id,
                status: WebGWASResultStatus::Done,
                message: None
            })
        }
    }
    info!("Cache miss");
    match validate_phenotype_definition(
        request.cohort_id,
        &request.phenotype_definition,
        &state.knowledge_base,
    ) {
        Ok(definition) => {
            let result = WebGWASResult {
                request_id: unique_id,
                status: WebGWASResultStatus::Queued,
                error_msg: None,
                local_result_file: None,
                local_zip_file: None,
            };

            let maybe_popped = state.results.lock().unwrap().insert(result);
            if let Some(popped) = maybe_popped {
                let result_file = popped.local_result_file.expect("Expected to find a local result file");
                let zip_file = popped.local_zip_file.expect("Expected to find a local zip file");
                let pop_result = remove_cache_entry(&state.db, &popped.request_id.to_string(), &result_file, &zip_file).await;
                if let Err(err) = pop_result {
                    error!("Error removing local files: {}", err)
                } else {
                    info!("Removed file without error")
                }
            }

            // Build the processed request
            let request = WebGWASRequestId {
                id: unique_id,
                raw_definition: request.phenotype_definition,
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
    match state.results.lock().unwrap().get(&request_id) {
        Some(result) => Json(result.clone()),
        None => Json(WebGWASResult {
            request_id,
            status: WebGWASResultStatus::Error,
            error_msg: Some(format!("No result found for request {}", request_id)),
            local_result_file: None,
            local_zip_file: None,
        }),
    }
}

fn get_client_ip<T>(request: &axum::http::Request<T>) -> String {
    // Try to get the IP from the X-Forwarded-For header
    if let Some(ip) = request
        .headers()
        .get("X-Forwarded-For")
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.split(',').next())
    {
        return ip.trim().to_string();
    }

    // If X-Forwarded-For is not available, try X-Real-IP
    if let Some(ip) = request
        .headers()
        .get("X-Real-IP")
        .and_then(|hv| hv.to_str().ok())
    {
        return ip.trim().to_string();
    }
    info!("No X-Forwarded-For or X-Real-IP header found, falling back to direct connection IP");
    info!("Headers: {:#?}", request.headers());

    // If neither header is available, fall back to the direct connection IP
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

async fn get_cached_results(db: &SqlitePool, cohort_id: i32, phenotype_definition: &str) -> Result<Option<CacheRow>> {
    let result = sqlx::query_as::<_, CacheRow>(
        "SELECT request_id, cohort_id, phenotype_definition, result_file, zip_file
        FROM cache
        WHERE cohort_id = $1 AND phenotype_definition = $2",
    )
    .bind(cohort_id)
    .bind(phenotype_definition)
    .fetch_optional(db)
    .await?;
    Ok(result)
}
