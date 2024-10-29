use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use http::HeaderName;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

use crate::AppState;

#[axum::debug_handler]
pub async fn download_stream(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<Uuid>,
) -> Result<Response, (StatusCode, String)> {
    let result = match state.results.lock().unwrap().get(&request_id) {
        Some(result) => result.clone(),
        None => return Err((StatusCode::NOT_FOUND, "Result not found".to_string())),
    };

    let file_path = result
        .local_zip_file
        .ok_or((StatusCode::NOT_FOUND, "Local path not found".to_string()))?;

    // Convert the full path to just the filename or relative path you want to expose
    let filename = file_path
        .file_name()
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Invalid filename".to_string(),
        ))?
        .to_str()
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Non-UTF8 filename".to_string(),
        ))?;
    info!("Downloading file {}", filename);

    // Create an X-Accel-Redirect header (for nginx internal redirect)
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-accel-redirect"),
        HeaderValue::from_str(&format!("/protected-downloads/{}", filename))
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
    );
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/zip"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{}\"", filename))
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
    );

    // Return a response with just the headers - nginx will handle the actual file serving
    let response = Response::builder()
        .status(StatusCode::OK)
        .extension(headers)
        .body(Body::empty())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(response)
}
