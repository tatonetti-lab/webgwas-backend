use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use std::sync::Arc;
use tokio::io::BufReader;
use tokio_util::io::ReaderStream;
use uuid::Uuid;

use crate::AppState;

#[axum::debug_handler]
pub async fn download_stream(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<Uuid>,
) -> Result<Response, (StatusCode, String)> {
    // Get metadata first before opening the file
    let result = match state.results.lock().unwrap().get(&request_id) {
        Some(result) => result.clone(),
        None => return Err((StatusCode::NOT_FOUND, "Result not found".to_string())),
    };

    let file_path = result
        .local_zip_file
        .ok_or((StatusCode::NOT_FOUND, "Local path not found".to_string()))?;

    // Get file metadata before opening the file
    let metadata = tokio::fs::metadata(&file_path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let headers = build_headers(&file_path, metadata.len()).await?;

    // Open file after headers are prepared
    let file = tokio::fs::File::open(&file_path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Use a larger buffer size for better performance
    let reader = BufReader::with_capacity(262_144, file); // 256KB buffer
    let stream = ReaderStream::new(reader);
    let body = Body::from_stream(stream);
    let response = Response::builder()
        .status(StatusCode::OK)
        .extension(headers)
        .body(body)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(response)
}

// Separated header building function with pre-calculated file size
async fn build_headers(
    download_file_path: &std::path::Path,
    file_size: u64,
) -> Result<HeaderMap, (StatusCode, String)> {
    let filename = download_file_path
        .file_name()
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Filename not found".to_string(),
            )
        })?
        .to_str()
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Filename not unicode".to_string(),
            )
        })?
        .to_string();

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/zip"),
    );
    headers.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&file_size.to_string())
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
    );
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    headers.insert(
        axum::http::header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{}\"", filename))
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
    );

    Ok(headers)
}
