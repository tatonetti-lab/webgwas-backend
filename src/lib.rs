use anyhow::{anyhow, bail, Context, Result};
use log::info;
use models::{CacheRow, Cohort, WebGWASResultStatus};
use phenotype_definitions::KnowledgeBase;
use polars::io::parquet::read::ParquetReader;
use polars::prelude::*;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{
    collections::HashMap,
    fs::File,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

pub mod config;
pub mod endpoints;
pub mod errors;
pub mod igwas;
pub mod models;
pub mod phenotype_definitions;
pub mod regression;
pub mod utils;
pub mod worker;

use crate::config::Settings;
use crate::models::{CohortData, Feature, PhenotypeFitQuality, WebGWASRequestId, WebGWASResult};

pub struct AppState {
    pub results_directory: PathBuf,
    pub settings: Settings,
    pub db: SqlitePool,
    pub knowledge_base: KnowledgeBase,
    pub cohort_id_to_data: Arc<Mutex<HashMap<i32, Arc<CohortData>>>>,
    pub fit_quality_reference: Arc<Vec<PhenotypeFitQuality>>,
    pub queue: Arc<Mutex<Vec<WebGWASRequestId>>>,
    pub results: Arc<Mutex<ResultsCache>>,
}

impl AppState {
    pub async fn new(settings: Settings) -> Result<Self> {
        let results_dir = Path::new(&settings.results_path);
        if std::fs::exists(results_dir)? {
            std::fs::create_dir_all(results_dir)?;
        }
        let data_dir = Path::new(&settings.data_path);
        if !std::fs::exists(data_dir)? {
            bail!("Data path does not exist");
        }
        let db_path = data_dir.join("webgwas.db").display().to_string();
        let db = connect_db(&db_path).await?;
        let cache = ResultsCache::new(settings.cache_capacity)
            .load(db.clone())
            .await?;

        // Remove all untracked files from the results directory
        let files_to_keep = cache
            .get_all_known_files()
            .into_iter()
            .collect::<HashSet<PathBuf>>();
        std::fs::read_dir(results_dir)?
            .map(|path| -> Result<()> {
                let path = path?.path();
                if files_to_keep.contains(&path) {
                    Ok(())
                } else {
                    Ok(std::fs::remove_file(path)?)
                }
            })
            .collect::<Result<Vec<()>>>()?;

        let cohort_id_to_data = load_cohort_data(&db, data_dir).await?;
        let knowledge_base = load_knowledge_base(&db).await?;
        let fit_quality_path = data_dir.join("fit_quality.parquet");
        let fit_quality_reference = load_fit_quality(&fit_quality_path).await?;

        let state = AppState {
            results_directory: results_dir.to_path_buf(),
            settings,
            db,
            knowledge_base,
            cohort_id_to_data: Arc::new(Mutex::new(cohort_id_to_data)),
            fit_quality_reference: Arc::new(fit_quality_reference),
            queue: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(Mutex::new(cache)),
        };
        info!("Finished initializing app state");
        Ok(state)
    }
}

async fn connect_db(db_path: &str) -> Result<SqlitePool> {
    let db = SqlitePoolOptions::new()
        .max_connections(20)
        .connect(db_path)
        .await
        .context(anyhow!("Failed to connect to database: {}", db_path))?;

    sqlx::query("PRAGMA journal_mode=WAL;").execute(&db).await?;
    sqlx::query("PRAGMA synchronous = NORMAL;")
        .execute(&db)
        .await?;
    sqlx::query("PRAGMA temp_store = MEMORY;")
        .execute(&db)
        .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS cache
        (
            request_id           TEXT PRIMARY KEY,
            phenotype_definition TEXT,
            result_file          TEXT,
            zip_file             TEXT,
            last_accessed        INTEGER NOT NULL,
            cohort_id REFERENCES cohort (id),
            UNIQUE(cohort_id, phenotype_definition) ON CONFLICT ABORT
        );",
    )
    .execute(&db)
    .await
    .context("Error creating table")?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache (last_accessed)")
        .execute(&db)
        .await
        .context("Error creating index")?;
    Ok(db)
}

async fn load_cohort_data(
    db: &SqlitePool,
    data_dir: &Path,
) -> Result<HashMap<i32, Arc<CohortData>>> {
    let cohort_id_to_data = sqlx::query_as::<_, Cohort>("SELECT * FROM cohort")
        .fetch_all(db)
        .await
        .context("Failed to fetch cohorts")?
        .into_iter()
        .map(|cohort| -> Result<CohortData> { CohortData::load(cohort, data_dir) })
        .collect::<Result<Vec<CohortData>>>()?
        .into_iter()
        .map(|cohort_data| {
            (
                cohort_data.cohort.id.expect("Cohort ID is missing"),
                Arc::new(cohort_data),
            )
        })
        .collect::<HashMap<i32, Arc<CohortData>>>();
    Ok(cohort_id_to_data)
}

async fn load_knowledge_base(db: &SqlitePool) -> Result<KnowledgeBase> {
    let fields = sqlx::query_as::<_, Feature>(
        "SELECT id, code, name, type as node_type, sample_size, cohort_id FROM feature",
    )
    .fetch_all(db)
    .await
    .context("Failed to fetch features")
    .unwrap();
    let kb = KnowledgeBase::new(fields);
    Ok(kb)
}

async fn load_fit_quality(fit_quality_path: &Path) -> Result<Vec<PhenotypeFitQuality>> {
    let fit_quality_file = File::open(fit_quality_path).context(anyhow!(
        "Failed to open fit quality file at {:?}",
        fit_quality_path
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
    Ok(fit_quality_reference)
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

    pub async fn load(self, db: SqlitePool) -> Result<Self> {
        info!("Loading cache");
        let mut result_self = self;
        let rows = sqlx::query_as::<_, CacheRow>(
            "SELECT request_id, cohort_id, phenotype_definition, result_file, zip_file 
            FROM cache 
            ORDER BY last_accessed DESC",
        )
        .fetch_all(&db)
        .await?;
        for row in rows.into_iter() {
            let result = WebGWASResult {
                request_id: Uuid::parse_str(&row.request_id)?,
                status: WebGWASResultStatus::Done,
                error_msg: None,
                local_result_file: Some(PathBuf::from_str(&row.result_file)?),
                local_zip_file: Some(PathBuf::from_str(&row.zip_file)?),
            };
            result_self.insert(result);
        }
        Ok(result_self)
    }

    pub fn insert(&mut self, result: WebGWASResult) -> Option<WebGWASResult> {
        let popped_value = if self.id_to_result.is_full() {
            let lru_key = *self.id_to_result.lru().unwrap();
            Some(self.id_to_result.remove(&lru_key).expect("No value found"))
        } else {
            None
        };
        self.id_to_result.insert(result.request_id, result);
        popped_value
    }

    pub fn get(&mut self, id: &Uuid) -> Option<&WebGWASResult> {
        self.id_to_result.get(id)
    }

    pub fn get_mut(&mut self, id: &Uuid) -> Option<&mut WebGWASResult> {
        self.id_to_result.get_mut(id)
    }

    pub fn get_all_known_files(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for (_, item) in self.id_to_result.iter() {
            paths.push(item.local_result_file.clone().expect("Expected local file"));
            paths.push(item.local_zip_file.clone().expect("Expected zip file"));
        }
        paths
    }
}

pub async fn remove_cache_entry(
    db: &SqlitePool,
    request_id: &str,
    local_file: &Path,
    zip_file: &Path,
) -> Result<()> {
    sqlx::query("DELETE FROM cache WHERE request_id = $1;")
        .bind(request_id)
        .execute(db)
        .await?;
    std::fs::remove_file(local_file)
        .context("Failed to remove local result file")
        .unwrap();
    std::fs::remove_file(zip_file)
        .context("Failed to remove local zip file")
        .unwrap();
    Ok(())
}
