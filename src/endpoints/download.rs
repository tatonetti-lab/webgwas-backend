use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderValue, StatusCode},
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

    // Build response with proper headers
    let mut response = Response::builder().status(StatusCode::OK);

    // Add headers directly to the response builder
    let headers = response.headers_mut().unwrap();

    // Add X-Accel-Redirect header for nginx
    headers.insert(
        HeaderName::from_static("x-accel-redirect"),
        HeaderValue::from_str(&format!("/protected-downloads/{}", filename))
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
    );

    // Ensure proper content type for zip files
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/zip"),
    );

    // Set content disposition with the correct filename
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"{}\"; filename*=UTF-8''{}",
            filename, filename
        ))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
    );

    // Build the response with an empty body (nginx will serve the actual file)
    let response = response
        .body(Body::empty())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(response)
}
