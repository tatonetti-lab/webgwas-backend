use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

pub struct WebGWASError(anyhow::Error);

impl IntoResponse for WebGWASError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, WebGWASError>`.
impl<E> From<E> for WebGWASError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
