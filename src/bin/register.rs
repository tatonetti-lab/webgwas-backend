use anyhow::{bail, Context, Result};
use clap::Parser;
use faer_ext::polars::polars_to_faer_f32;
use log::{error, info};
use polars::prelude::*;
use rusqlite::OptionalExtension;
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use webgwas_backend::{
    models::{Cohort, CohortMetadata},
    regression::regress_mat,
};

fn main() {
    env_logger::init();
    let args = Cli::parse();
    let mut app_state = LocalAppState::new(args.cohort_name.clone())
        .context("Failed to initialize app state")
        .unwrap();
    let result = app_state.register_cohort(
        args.pheno_file,
        args.covar_file,
        args.gwas_files,
        args.feature_info_file,
        args.gwas_spec,
        args.pheno_spec,
    );
    match result {
        Ok(_) => {
            info!("Successfully registered cohort '{}'", args.cohort_name);
        }
        Err(err) => {
            app_state
                .cleanup()
                .context("Failed to cleanup app state")
                .unwrap();
            error!("Failed to register cohort: {}", err)
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Cohort name (mandatory)
    #[arg(long, required = true)]
    cohort_name: String,

    // Phenotype file arguments
    #[command(flatten)]
    pheno_file: PhenoFile,

    // Covariate file arguments (optional)
    #[command(flatten)]
    covar_file: CovarFile,

    /// GWAS files
    #[arg(long = "gwas-files", num_args = 1..)]
    gwas_files: Vec<PathBuf>,

    /// File giving information about each feature
    #[command(flatten)]
    feature_info_file: FeatureInfoFile,

    // Options for processing GWAS files
    #[command(flatten)]
    gwas_spec: GWASOptions,

    // Options for processing phenotype files
    #[command(flatten)]
    pheno_spec: PhenoOptions,
}

#[derive(Parser, Debug)]
pub struct PhenoFile {
    /// Phenotype file path (mandatory)
    #[arg(long = "pheno-path", required = true)]
    pheno_path: PathBuf,

    /// Phenotype file separator
    #[arg(long = "pheno-separator", default_value_t = b'\t')]
    pheno_separator: u8,

    /// Phenotype file person ID column (mandatory)
    #[arg(long = "pheno-person-id-column", default_value = "eid")]
    pheno_person_id_column: String,

    /// Phenotype file ignore columns
    #[arg(long = "pheno-ignore-columns")]
    pheno_ignore_columns: Option<Vec<String>>,
}

#[derive(Parser, Debug)]
pub struct CovarFile {
    /// Covariate file path
    #[arg(long = "covar-path", required = true)]
    covar_path: PathBuf,

    /// Covariate file separator (requires covar-path)
    #[arg(
        long = "covar-separator",
        requires = "covar_path",
        default_value_t = b'\t'
    )]
    covar_separator: u8,

    /// Covariate file person ID column (requires covar-path)
    #[arg(
        long = "covar-person-id-column",
        requires = "covar_path",
        default_value = "eid"
    )]
    covar_person_id_column: String,

    /// Covariate file ignore columns (requires covar-path)
    #[arg(long = "covar-ignore-columns", requires = "covar_path")]
    covar_ignore_columns: Option<Vec<String>>,
}

#[derive(Parser, Debug)]
pub struct FeatureInfoFile {
    /// Feature info file path (mandatory)
    #[arg(long = "feature-info-path", required = true)]
    feature_info_path: PathBuf,

    /// Feature info file separator
    #[arg(long = "feature-info-separator", default_value_t = b'\t')]
    feature_info_separator: u8,
}

#[derive(Parser, Debug)]
pub struct GWASOptions {
    /// GWAS file separator
    #[arg(long = "gwas-separator", default_value_t = b'\t')]
    pub gwas_separator: u8,

    /// GWAS variant ID column
    #[arg(long = "variant-id-column", default_value = "ID")]
    pub variant_id_column: String,

    /// GWAS beta column
    #[arg(long = "beta-column", default_value = "BETA")]
    pub beta_column: String,

    /// GWAS standard error column
    #[arg(long = "std-err-column", default_value = "SE")]
    pub std_err_column: String,

    /// GWAS sample size column
    #[arg(long = "sample-size-column", default_value = "OBS_CT")]
    pub sample_size_column: String,

    /// Keep n variants
    #[arg(long = "keep-n-variants")]
    pub keep_n_variants: Option<i32>,
}

#[derive(Parser, Debug)]
pub struct PhenoOptions {
    /// k-anonymity
    #[arg(long = "k-anonymity", default_value = "10")]
    k_anonymity: i32,

    /// Keep n samples
    #[arg(long = "keep-n-samples")]
    keep_n_samples: Option<i32>,

    /// Mean center
    #[arg(long = "mean-center", default_value = "true")]
    mean_center: bool,
}

pub struct LocalAppState {
    pub root_directory: PathBuf,
    pub cohort_directory: Option<PathBuf>,
    pub db: rusqlite::Connection,
    pub feature_sets: FeatureSets,
    pub metadata: CohortMetadata,
}

#[derive(Debug, Default)]
pub struct FeatureSets {
    pub phenotype_file: HashSet<String>,
    pub gwas_files: HashSet<String>,
    pub info_file: HashSet<String>,
    pub final_features: HashSet<String>,
}

impl LocalAppState {
    pub fn new(cohort_name: String) -> Result<Self> {
        let home = std::env::var("HOME").expect("Failed to read $HOME");
        let root_directory = PathBuf::from(home).join("webgwas");
        if !root_directory.exists() {
            std::fs::create_dir_all(&root_directory)?;
        }
        let sqlite_db_path = root_directory.join("webgwas.db");
        let db = initialize_database(&sqlite_db_path)?;
        let feature_sets = FeatureSets::default();
        let metadata = CohortMetadata {
            cohort_id: None,
            cohort_name,
            num_covar: None,
        };
        Ok(Self {
            root_directory,
            cohort_directory: None,
            db,
            feature_sets,
            metadata,
        })
    }

    pub fn register_cohort(
        &mut self,
        pheno_file_spec: PhenoFile,
        covar_file_spec: CovarFile,
        gwas_files: Vec<PathBuf>,
        feature_info_file: FeatureInfoFile,
        gwas_options: GWASOptions,
        pheno_options: PhenoOptions,
    ) -> Result<()> {
        info!("Registering cohort {}", self.metadata.cohort_name);
        self.check_cohort_does_not_exist()?;
        self.find_usable_features(
            &pheno_file_spec,
            &covar_file_spec,
            &gwas_files,
            &feature_info_file,
        )?;
        // TODO: Process the phenotype/covariate files
        self.process_phenotypes_covariates(&pheno_file_spec, &covar_file_spec, pheno_options)?;

        // TODO: Process the GWAS files
        self.process_gwas_files(&gwas_files, gwas_options)?;

        // TODO: Process the feature info file
        self.process_feature_info_file(&feature_info_file)?;

        error!("Hit end of current implementation");
        bail!("Unimplement register_cohort")
    }

    pub fn check_cohort_does_not_exist(&mut self) -> Result<()> {
        check_cohort_not_in_db(&self.metadata.cohort_name, &self.db)?;
        let normalized_cohort_name = normalize_cohort_name(&self.metadata.cohort_name);
        let cohort_directory = self
            .root_directory
            .join("cohorts")
            .join(&normalized_cohort_name);
        if cohort_directory.exists() {
            bail!("Cohort {} already exists", normalized_cohort_name);
        }
        std::fs::create_dir_all(&cohort_directory)?;
        self.cohort_directory = Some(cohort_directory);
        Ok(())
    }

    pub fn find_usable_features(
        &mut self,
        pheno_file_spec: &PhenoFile,
        covar_file_spec: &CovarFile,
        gwas_files: &[PathBuf],
        feature_info_file: &FeatureInfoFile,
    ) -> Result<()> {
        // Find features in the phenotype file and count the number of covariates
        info!("Registering phenotypes and covariates");
        self.register_phenotypes_covariates(pheno_file_spec, covar_file_spec)?;

        // Find features in the GWAS files
        info!("Registering GWAS files");
        self.register_gwas_files(gwas_files)?;

        // Find features in the feature info file
        info!("Registering feature info file");
        self.register_feature_info(feature_info_file)?;

        // Find the shared features
        self.feature_sets.final_features = self
            .feature_sets
            .phenotype_file
            .intersection(&self.feature_sets.gwas_files)
            .cloned()
            .collect();
        self.feature_sets.final_features = self
            .feature_sets
            .final_features
            .intersection(&self.feature_sets.info_file)
            .cloned()
            .collect();
        let num_final_features = self.feature_sets.final_features.len();
        info!("Found {} shared features", num_final_features);
        Ok(())
    }

    pub fn register_phenotypes_covariates(
        &mut self,
        pheno_file_spec: &PhenoFile,
        covar_file_spec: &CovarFile,
    ) -> Result<()> {
        let num_covar = get_num_covar(covar_file_spec)?;
        self.metadata.num_covar = Some(num_covar);

        // Get all the phenotype names
        let mut ignore_columns = pheno_file_spec
            .pheno_ignore_columns
            .clone()
            .unwrap_or_default();
        ignore_columns.push(pheno_file_spec.pheno_person_id_column.clone());
        let phenotype_names = read_delim_header(
            &pheno_file_spec.pheno_path,
            pheno_file_spec.pheno_separator,
            Some(ignore_columns),
        )?;
        self.feature_sets.phenotype_file = phenotype_names.into_iter().collect();
        Ok(())
    }

    pub fn register_gwas_files(&mut self, gwas_files: &[PathBuf]) -> Result<()> {
        for gwas_file in gwas_files {
            let feature_name = get_gwas_feature_name(gwas_file);
            match self.feature_sets.gwas_files.insert(feature_name) {
                true => {}
                false => bail!("GWAS file {} already exists", gwas_file.display()),
            }
        }
        Ok(())
    }

    pub fn register_feature_info(&mut self, feature_info_file: &FeatureInfoFile) -> Result<()> {
        let parse_opts =
            CsvParseOptions::default().with_separator(feature_info_file.feature_info_separator);
        let codes = CsvReadOptions::default()
            .with_columns(Some(Arc::new(["code".into()])))
            .with_parse_options(parse_opts)
            .try_into_reader_with_file_path(Some(feature_info_file.feature_info_path.clone()))
            .context("Failed to set feature info file as path")?
            .finish()?["code"]
            .str()?
            .into_iter()
            .map(|x_opt| x_opt.unwrap().to_string())
            .collect::<Vec<String>>();
        for code in codes.iter() {
            match self.feature_sets.info_file.insert(code.clone()) {
                true => {}
                false => bail!("Feature info file contains duplicate code {}", code),
            }
        }
        Ok(())
    }

    pub fn process_phenotypes_covariates(
        &mut self,
        pheno_file_spec: &PhenoFile,
        covar_file_spec: &CovarFile,
        pheno_options: PhenoOptions,
    ) -> Result<()> {
        let pheno_df = CsvReadOptions::default()
            .with_parse_options(
                CsvParseOptions::default().with_separator(pheno_file_spec.pheno_separator),
            )
            .try_into_reader_with_file_path(Some(pheno_file_spec.pheno_path.clone()))
            .context("Failed to set phenotype file as path")?
            .finish()?;
        let covar_df = CsvReadOptions::default()
            .with_parse_options(
                CsvParseOptions::default().with_separator(covar_file_spec.covar_separator),
            )
            .try_into_reader_with_file_path(Some(covar_file_spec.covar_path.clone()))
            .context("Failed to set covariate file as path")?
            .finish()?;
        info!("Merging phenotype and covariate files");
        let merged_df = pheno_df.join(
            &covar_df,
            [pheno_file_spec.pheno_person_id_column.as_str()],
            [covar_file_spec.covar_person_id_column.as_str()],
            JoinArgs::new(JoinType::Inner),
        )?;
        let pheno_cols = pheno_df
            .drop(pheno_file_spec.pheno_person_id_column.as_str())?
            .get_column_names()
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        let y_df = merged_df.select(pheno_cols)?.lazy();
        let y = polars_to_faer_f32(y_df)?;
        let covar_cols = covar_df
            .drop(covar_file_spec.covar_person_id_column.as_str())?
            .get_column_names()
            .into_iter()
            .map(|x| col(x.to_string()))
            .collect::<Vec<Expr>>();
        let x_lazy = merged_df
            .lazy()
            .select(covar_cols)
            .with_column(lit(1.0).alias("intercept"));
        let x = polars_to_faer_f32(x_lazy)?;
        info!("Regressing phenotype against covariates");
        let start = Instant::now();
        let beta = regress_mat(&y, &x);
        let duration = start.elapsed();
        info!("Regression took {:?}", duration);

        info!("Residualizing phenotypes");
        let start = Instant::now();
        let y_res = y - &x * &beta;
        let duration = start.elapsed();
        info!("Residualizing phenotypes took {:?}", duration);

        info!("Calculating covariance matrix");
        let start = Instant::now();
        let covar = x.transpose() * &x;
        let duration = start.elapsed();
        info!("Calculating covariance matrix took {:?}", duration);

        bail!("Unimplement process_phenotypes_covariates")
    }

    pub fn process_gwas_files(
        &mut self,
        gwas_files: &[PathBuf],
        gwas_options: GWASOptions,
    ) -> Result<()> {
        bail!("Unimplement process_gwas_files")
    }

    pub fn process_feature_info_file(&mut self, feature_info_file: &FeatureInfoFile) -> Result<()> {
        bail!("Unimplement process_feature_info_file")
    }

    pub fn cleanup(&self) -> Result<()> {
        self.db.execute(
            "DELETE FROM cohort WHERE name = ?",
            [&self.metadata.cohort_name],
        )?;
        self.db.execute(
            "DELETE FROM phenotype WHERE cohort_id = ?",
            [&self.metadata.cohort_id],
        )?;
        let dir = self
            .cohort_directory
            .clone()
            .expect("Cohort directory not set");
        std::fs::remove_dir_all(dir)?;
        info!("Cleaned up all cohort data");
        Ok(())
    }
}

pub fn normalize_cohort_name(cohort_name: &str) -> String {
    cohort_name.to_lowercase().replace(" ", "_")
}

pub fn initialize_database(path: &Path) -> Result<rusqlite::Connection> {
    let db = rusqlite::Connection::open(path)?;
    db.execute(
        "CREATE TABLE IF NOT EXISTS cohort (
            id INTEGER NOT NULL,
            name VARCHAR NOT NULL,
            root_directory VARCHAR NOT NULL,
            num_covar INTEGER NOT NULL,
            PRIMARY KEY (id),
            UNIQUE (name),
            UNIQUE (root_directory)
        )",
        (),
    )?;
    db.execute(
        "CREATE TABLE IF NOT EXISTS phenotype (
                id INTEGER NOT NULL,
                code VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                type VARCHAR(4) NOT NULL,
                cohort_id INTEGER,
                PRIMARY KEY (id),
                CONSTRAINT unique_feature UNIQUE (cohort_id, code, name),
                FOREIGN KEY(cohort_id) REFERENCES cohort (id)
        )",
        (),
    )?;
    Ok(db)
}

/// Check if the cohort already exists in the database
pub fn check_cohort_not_in_db(cohort_name: &str, db: &rusqlite::Connection) -> Result<()> {
    let mut stmt = db.prepare(
        "SELECT id, name, root_directory, num_covar
            FROM cohort WHERE name = ?",
    )?;
    let existing_cohort = stmt
        .query_row([cohort_name], |row| {
            Ok(Cohort {
                id: row.get(0)?,
                name: row.get(1)?,
                root_directory: row.get(2)?,
                num_covar: row.get(3)?,
            })
        })
        .optional()?;
    if let Some(cohort) = existing_cohort {
        bail!("Cohort {} already exists", cohort.name);
    }
    Ok(())
}

/// Read the number of covariates
pub fn get_num_covar(covar_file_spec: &CovarFile) -> Result<i32> {
    let mut ignore_columns = covar_file_spec
        .covar_ignore_columns
        .clone()
        .unwrap_or_default();
    ignore_columns.push(covar_file_spec.covar_person_id_column.clone());
    let header = read_delim_header(
        &covar_file_spec.covar_path,
        covar_file_spec.covar_separator,
        Some(ignore_columns),
    )?;
    let num_covar = header.len() as i32;
    Ok(num_covar)
}

/// Read the header of a delimited file
pub fn read_delim_header(
    path: &Path,
    separator: u8,
    ignore_columns: Option<Vec<String>>,
) -> Result<Vec<String>> {
    let parse_opts = CsvParseOptions::default().with_separator(separator);
    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .with_n_rows(Some(0))
        .with_parse_options(parse_opts)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))
        .context("Failed to set covariate file as path")?
        .finish()
        .context("Failed to read covariate file header")?;
    if let Some(ignore_columns) = ignore_columns {
        df = df.drop_many(ignore_columns);
    }
    let names = df
        .get_column_names()
        .iter()
        .map(|x| x.to_string())
        .collect();
    Ok(names)
}

/// Extract the phenotype name from a GWAS file path (e.g. /path/to/file.tsv.zst -> file)
pub fn get_gwas_feature_name(gwas_file: &Path) -> String {
    let file_name = gwas_file.file_name().unwrap().to_str().unwrap();
    let feature_name = file_name.split('.').next().unwrap();
    feature_name.to_string()
}
