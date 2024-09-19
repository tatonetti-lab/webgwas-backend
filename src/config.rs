use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub enum LogLevel {
    #[serde(alias = "DEBUG")]
    Debug,
    #[serde(alias = "INFO")]
    Info,
    #[serde(alias = "WARN")]
    Warn,
    #[serde(alias = "ERROR")]
    Error,
}

#[derive(Deserialize, Debug)]
pub struct Settings {
    pub cache_capacity: usize,
    pub s3_region: String,
    pub s3_bucket: String,
    pub s3_result_path: String,
}

impl Settings {
    pub fn read_file(toml_path: &str) -> Result<Self> {
        let contents = std::fs::read_to_string(toml_path)?;
        let settings = toml::from_str::<Settings>(&contents)?;
        Ok(settings)
    }
}
