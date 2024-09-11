extern crate blas_src;

use anyhow::{anyhow, Context, Result};
use aws_config::Region;
use aws_sdk_s3::Client;
use log::debug;
use phenotype_definitions::KnowledgeBase;
use polars::io::{parquet::read::ParquetReader, SerReader};
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
pub mod models;
pub mod phenotype_definitions;
pub mod regression;
pub mod worker;

use crate::config::Settings;
use crate::models::{CohortInfo, Feature, PhenotypeFitQuality, WebGWASRequestId, WebGWASResult};

pub struct AppState {
    pub settings: Settings,
    pub db: SqlitePool,
    pub s3_client: aws_sdk_s3::Client,
    pub knowledge_base: KnowledgeBase,
    pub cohort_id_to_info: HashMap<i32, Arc<CohortInfo>>,
    pub fit_quality_reference: Arc<Vec<PhenotypeFitQuality>>,
    pub queue: Arc<Mutex<Vec<WebGWASRequestId>>>,
    pub results: Arc<Mutex<HashMap<Uuid, WebGWASResult>>>,
}

impl AppState {
    pub async fn new(settings: Settings) -> Result<Self> {
        debug!("Initializing database");
        let db = SqlitePool::connect(&settings.sqlite_db_path)
            .await
            .context(anyhow!(
                "Failed to connect to database: {}",
                settings.sqlite_db_path
            ))?;

        debug!("Fetching features");
        let fields = sqlx::query_as::<_, Feature>(
            "SELECT id, code, name, type as node_type, cohort_id FROM feature",
        )
        .fetch_all(&db)
        .await
        .context("Failed to fetch features")
        .unwrap();
        let kb = KnowledgeBase::new(fields);

        debug!("Loading cohorts");
        let mut cohort_id_to_info = HashMap::new();
        for cohort_path in settings.cohort_paths.iter() {
            let path = Path::new(cohort_path);
            let info = CohortInfo::load(path)
                .context(anyhow!("Failed to load cohort info for {}", cohort_path))?;
            cohort_id_to_info.insert(info.cohort_id, Arc::new(info));
            debug!("Loaded cohort info for {}", cohort_path);
        }

        debug!("Initializing S3 client");
        let region = Region::new(settings.s3_region.clone());
        let shared_config = aws_config::from_env().region(region).load().await;
        let s3_client = Client::new(&shared_config);

        debug!("Loading fit quality reference");
        let fit_quality_file = File::open(&settings.fit_quality_file)?;
        let fit_quality_df = ParquetReader::new(fit_quality_file).finish()?;
        let fit_quality_reference = fit_quality_df
            .column("gwas_r2")?
            .f32()?
            .iter()
            .zip(fit_quality_df.column("phenotype_r2")?.f32()?.iter())
            .map(|(x, y)| {
                Some(PhenotypeFitQuality {
                    phenotype_fit_quality: x?,
                    gwas_fit_quality: y?,
                })
            })
            .collect::<Option<Vec<PhenotypeFitQuality>>>()
            .context("Failed to load fit quality reference")?;

        let state = AppState {
            settings,
            db,
            s3_client,
            knowledge_base: kb,
            cohort_id_to_info,
            fit_quality_reference: Arc::new(fit_quality_reference),
            queue: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(Mutex::new(HashMap::new())),
        };
        debug!("Finished initializing app state");
        Ok(state)
    }
}
