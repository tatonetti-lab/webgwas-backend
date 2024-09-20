use anyhow::{anyhow, Context, Result};
use aws_config::Region;
use aws_sdk_s3::Client;
use log::info;
use models::Cohort;
use phenotype_definitions::KnowledgeBase;
use polars::io::parquet::read::ParquetReader;
use polars::prelude::*;
use sqlx::SqlitePool;
use std::path::Path;
use std::{
    collections::HashMap,
    fs::File,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

pub mod config;
pub mod errors;
pub mod igwas;
pub mod models;
pub mod phenotype_definitions;
pub mod regression;
pub mod worker;

use crate::config::Settings;
use crate::models::{CohortData, Feature, PhenotypeFitQuality, WebGWASRequestId, WebGWASResult};

pub struct AppState {
    pub settings: Settings,
    pub db: SqlitePool,
    pub s3_client: aws_sdk_s3::Client,
    pub knowledge_base: KnowledgeBase,
    pub cohort_id_to_data: Arc<Mutex<HashMap<i32, Arc<CohortData>>>>,
    pub fit_quality_reference: Arc<Vec<PhenotypeFitQuality>>,
    pub queue: Arc<Mutex<Vec<WebGWASRequestId>>>,
    pub results: Arc<Mutex<ResultsCache>>,
}

impl AppState {
    pub async fn new(settings: Settings) -> Result<Self> {
        info!("Initializing database");
        let home = std::env::var("HOME").expect("Failed to read $HOME");
        let root = Path::new(&home).join("webgwas");
        let db_path = root.join("webgwas.db").display().to_string();
        let db = SqlitePool::connect(&db_path)
            .await
            .context(anyhow!("Failed to connect to database: {}", db_path))?;

        info!("Loading cohorts");
        let cohort_id_to_data = sqlx::query_as::<_, Cohort>("SELECT * FROM cohort")
            .fetch_all(&db)
            .await
            .context("Failed to fetch cohorts")?
            .into_iter()
            .map(|cohort| -> Result<CohortData> { CohortData::load(cohort, &root) })
            .collect::<Result<Vec<CohortData>>>()?
            .into_iter()
            .map(|cohort_data| {
                (
                    cohort_data.cohort.id.expect("Cohort ID is missing"),
                    Arc::new(cohort_data),
                )
            })
            .collect::<HashMap<i32, Arc<CohortData>>>();

        info!("Fetching features");
        let fields = sqlx::query_as::<_, Feature>(
            "SELECT id, code, name, type as node_type, sample_size, cohort_id FROM feature",
        )
        .fetch_all(&db)
        .await
        .context("Failed to fetch features")
        .unwrap();
        let kb = KnowledgeBase::new(fields);

        info!("Initializing S3 client");
        let region = Region::new(settings.s3_region.clone());
        let shared_config = aws_config::from_env().region(region).load().await;
        let s3_client = Client::new(&shared_config);

        info!("Loading fit quality reference");
        let fit_quality_path = root.join("fit_quality.parquet");
        let fit_quality_file = File::open(&fit_quality_path).context(anyhow!(
            "Failed to open fit quality file at {}",
            fit_quality_path.display()
        ))?;
        let fit_quality_df = ParquetReader::new(fit_quality_file).finish()?;
        let fit_quality_reference = fit_quality_df
            .column("gwas_r2")?
            .f32()?
            .iter()
            .zip(fit_quality_df.column("phenotype_r2")?.f32()?.iter())
            .map(|(g, p)| {
                Some(PhenotypeFitQuality {
                    phenotype_fit_quality: p?,
                    gwas_fit_quality: g?,
                })
            })
            .collect::<Option<Vec<PhenotypeFitQuality>>>()
            .context("Failed to load fit quality reference")?;

        let results = Arc::new(Mutex::new(ResultsCache::new(settings.cache_capacity)));

        let state = AppState {
            settings,
            db,
            s3_client,
            knowledge_base: kb,
            cohort_id_to_data: Arc::new(Mutex::new(cohort_id_to_data)),
            fit_quality_reference: Arc::new(fit_quality_reference),
            queue: Arc::new(Mutex::new(Vec::new())),
            results,
        };
        info!("Finished initializing app state");
        Ok(state)
    }
}

pub struct ResultsCache {
    id_to_result: hashlru::Cache<Uuid, WebGWASResult>,
}

impl ResultsCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            id_to_result: hashlru::Cache::new(capacity),
        }
    }

    pub fn insert(&mut self, result: WebGWASResult) {
        if self.id_to_result.is_full() {
            let lru_key = *self.id_to_result.lru().unwrap();
            let lru_value = self.id_to_result.remove(&lru_key).expect("No value found");
            let file_path = lru_value.local_result_file.expect("No local result file");
            std::fs::remove_file(file_path)
                .context("Failed to remove local result file")
                .unwrap();
        }
        self.id_to_result.insert(result.request_id, result);
    }

    pub fn get(&mut self, id: &Uuid) -> Option<&WebGWASResult> {
        self.id_to_result.get(id)
    }
}
