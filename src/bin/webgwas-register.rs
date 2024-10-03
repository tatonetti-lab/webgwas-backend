use anyhow::{bail, Context, Result};
use clap::Parser;
use faer::Mat;
use faer_ext::polars::polars_to_faer_f32;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use itertools::izip;
use log::{error, info, warn};
use mdav::mdav::mdav;
use polars::lazy::dsl::{mean_horizontal, min_horizontal};
use polars::prelude::*;
use rusqlite::{params, OptionalExtension};
use std::ops::Sub;
use std::str::FromStr;
use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::fmt::time::ChronoLocal;
use webgwas_backend::models::NodeType;

use webgwas_backend::{
    models::Cohort,
    regression::{
        compute_covariance, compute_left_inverse, residualize_covariates, transpose_vec_vec,
    },
};

fn main() {
    let subscriber = tracing_subscriber::fmt()
        .with_target(false)
        .with_span_events(FmtSpan::CLOSE)
        .with_timer(ChronoLocal::new("%r".to_string()));
    subscriber.init();

    let args = Cli::parse();
    let mut app_state = LocalAppState::new(args.cohort_name.clone(), args.overwrite)
        .context("Failed to initialize app state")
        .unwrap();
    let result = app_state.register_cohort(
        args.pheno_file,
        args.covar_file,
        args.gwas_files,
        args.feature_info_file,
        args.gwas_spec,
        args.pheno_spec,
        args.variant_info_spec,
    );
    match result {
        Ok(_) => {
            app_state
                .check_result()
                .context("Final check failed")
                .unwrap();
            info!("Successfully registered cohort '{}'", args.cohort_name);
        }
        Err(err) => {
            error!("Failed to register cohort: {}", err);
            if !args.no_cleanup {
                info!("Cleaning up");
                app_state
                    .cleanup()
                    .context("Failed to cleanup app state")
                    .unwrap();
            }
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

    // Options for processing variant info files
    #[command(flatten)]
    variant_info_spec: VariantInfoOptions,

    // Overwrite existing cohort
    #[arg(long = "overwrite", default_value_t = false)]
    overwrite: bool,

    // Don't clean up if an error occurs
    #[arg(long = "no-cleanup", default_value_t = false)]
    no_cleanup: bool,
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

    /// GWAS A1 column
    #[arg(long = "a1-column", default_value = "A1")]
    pub a1_column: String,

    /// GWAS A2 column
    #[arg(long = "a2-column", default_value = "OMITTED")]
    pub a2_column: String,

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
    pub keep_n_variants: Option<usize>,
}

#[derive(Parser, Debug)]
pub struct PhenoOptions {
    /// k-anonymity
    #[arg(long = "k-anonymity", default_value = "10")]
    k_anonymity: usize,

    /// Keep n samples
    #[arg(long = "keep-n-samples")]
    keep_n_samples: Option<usize>,

    /// Plink offset
    #[arg(long = "plink-offset", default_value_t = false)]
    plink_offset: bool,

    /// Min sample size
    #[arg(long = "min-sample-size")]
    min_sample_size: Option<usize>,
}

#[derive(Parser, Debug)]
pub struct VariantInfoOptions {
    /// Variant info file path (mandatory)
    #[arg(long = "variant-info-path")]
    variant_info_path: Option<PathBuf>,

    /// Variant info file separator
    #[arg(long = "variant-info-separator", default_value_t = b'\t')]
    variant_info_separator: u8,

    /// Variant info variant ID column
    #[arg(long = "variant-info-variant-id-column", default_value = "variant_id")]
    variant_info_variant_id_column: String,

    /// Variant info columns
    #[arg(long = "variant-info-columns")]
    variant_info_columns: Option<Vec<String>>,
}

pub struct LocalAppState {
    pub root_directory: PathBuf,
    pub cohort_directory: PathBuf,
    pub db: rusqlite::Connection,
    pub feature_sets: FeatureSets,
    pub metadata: Cohort,
    pub feature_info_map: HashMap<String, FeatureInfo>,
    pub overwrite: bool,
}

#[derive(Debug, Default)]
pub struct FeatureSets {
    pub phenotype_file: HashSet<String>,
    pub gwas_files: HashSet<String>,
    pub info_file: HashSet<String>,
    pub gwas_feature_to_file: HashMap<String, PathBuf>,
    pub final_features: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct FeatureInfo {
    pub code: String,
    pub name: String,
    pub type_: Option<NodeType>,
    pub sample_size: Option<f32>,
    pub partial_variance: Option<f32>,
}

impl LocalAppState {
    pub fn new(cohort_name: String, overwrite: bool) -> Result<Self> {
        let home = std::env::var("HOME").expect("Failed to read $HOME");
        let root_directory = PathBuf::from(home).join("webgwas");
        if !root_directory.exists() {
            std::fs::create_dir_all(&root_directory)?;
        }
        let sqlite_db_path = root_directory.join("webgwas.db");
        let db = initialize_database(&sqlite_db_path)?;
        let feature_sets = FeatureSets::default();
        let normalized_cohort_name = normalize_cohort_name(&cohort_name);
        let cohort_directory = root_directory.join("cohorts").join(&normalized_cohort_name);
        let metadata = Cohort {
            id: None,
            name: cohort_name,
            normalized_name: normalized_cohort_name,
            num_covar: None,
        };
        Ok(Self {
            root_directory,
            cohort_directory,
            db,
            feature_sets,
            metadata,
            feature_info_map: HashMap::new(),
            overwrite,
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
        variant_info_options: VariantInfoOptions,
    ) -> Result<()> {
        info!("Registering cohort {}", self.metadata.name);
        match self.check_cohort_does_not_exist() {
            Ok(_) => {
                std::fs::create_dir_all(&self.cohort_directory)?;
            }
            Err(_) => {
                if self.overwrite {
                    warn!("Cohort {} already exists, overwriting", self.metadata.name);
                    let _ = std::fs::remove_dir_all(&self.cohort_directory);
                    std::fs::create_dir_all(&self.cohort_directory)?;
                    let _ = self.delete_cohort_from_database();
                } else {
                    bail!("Cohort {} already exists", self.metadata.name);
                }
            }
        }
        self.find_usable_features(
            &pheno_file_spec,
            &covar_file_spec,
            &gwas_files,
            &feature_info_file,
        )?;
        self.process_feature_info_file(&feature_info_file)?;
        self.process_phenotypes_covariates(&pheno_file_spec, &covar_file_spec, pheno_options)?;
        self.process_gwas_files(gwas_options, variant_info_options)?;
        let cohort_row = {
            let _span = info_span!("Writing cohort row to the database").entered();
            let cohort_row = Cohort {
                id: None,
                name: self.metadata.name.clone(),
                normalized_name: self.metadata.normalized_name.clone(),
                num_covar: Some(self.metadata.num_covar.expect("num_covar is missing")),
            };
            self.db.execute(
                "INSERT INTO cohort (name, normalized_name, num_covar) VALUES (?1, ?2, ?3)",
                params![
                    &cohort_row.name,
                    &cohort_row.normalized_name,
                    &cohort_row.num_covar.unwrap(),
                ],
            )?;
            cohort_row
        };

        let _span = info_span!("Writing feature info rows to the database").entered();
        let cohort_id: i32 = self.db.query_row(
            "SELECT id FROM cohort WHERE name = ?1",
            [&cohort_row.name],
            |row| row.get(0),
        )?;
        let tx = self.db.transaction()?;
        for feature_info_row in self.feature_info_map.values() {
            tx.execute(
                "INSERT INTO feature (code, name, type, sample_size, cohort_id) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    &feature_info_row.code,
                    &feature_info_row.name,
                    &feature_info_row.type_.unwrap().to_string(),
                    &feature_info_row.sample_size.unwrap(),
                    &cohort_id,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn check_cohort_does_not_exist(&mut self) -> Result<()> {
        check_cohort_not_in_db(&self.metadata.name, &self.db)?;
        if self.cohort_directory.exists() {
            bail!("Cohort {} already exists", self.metadata.name);
        }
        Ok(())
    }

    pub fn delete_cohort_from_database(&mut self) -> Result<()> {
        let _span = info_span!("Deleting cohort from the database").entered();
        let cohort_id: i32 = match self.db.query_row(
            "SELECT id FROM cohort WHERE name = ?1",
            [&self.metadata.name],
            |row| row.get(0),
        ) {
            Ok(id) => id,
            Err(err) => {
                warn!("Cohort was not found in the database: {}", err);
                return Ok(());
            }
        };
        let tx = self.db.transaction()?;
        tx.execute("DELETE FROM feature WHERE cohort_id = ?1", [&cohort_id])?;
        tx.execute("DELETE FROM cohort WHERE name = ?1", [&self.metadata.name])?;
        tx.commit()?;
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
        self.register_phenotypes_covariates(pheno_file_spec, covar_file_spec)?;
        info!(
            "Found {} phenotypes e.g. {}",
            self.feature_sets.phenotype_file.len(),
            self.feature_sets
                .phenotype_file
                .iter()
                .take(2)
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        );

        // Find features in the GWAS files
        self.register_gwas_files(gwas_files)?;
        info!(
            "Found {} GWAS files e.g. {}",
            self.feature_sets.gwas_files.len(),
            self.feature_sets
                .gwas_files
                .iter()
                .take(2)
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        );

        // Find features in the feature info file
        self.register_feature_info(feature_info_file)?;
        info!(
            "Found {} feature info rows e.g. {}",
            self.feature_sets.info_file.len(),
            self.feature_sets
                .info_file
                .iter()
                .take(2)
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        );

        // Find the shared features
        self.feature_sets.final_features = self
            .feature_sets
            .phenotype_file
            .intersection(&self.feature_sets.gwas_files)
            .cloned()
            .collect();
        self.feature_sets
            .final_features
            .retain(|x| self.feature_sets.info_file.contains(x));
        self.feature_sets.final_features.sort();
        let num_final_features = self.feature_sets.final_features.len();
        info!("Found {} shared features", num_final_features);
        if num_final_features == 0 {
            bail!("No shared features found");
        }
        Ok(())
    }

    pub fn register_phenotypes_covariates(
        &mut self,
        pheno_file_spec: &PhenoFile,
        covar_file_spec: &CovarFile,
    ) -> Result<()> {
        let _span = info_span!("Registering phenotypes and covariates").entered();
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
        let _span = info_span!("Registering GWAS files").entered();
        for gwas_file in gwas_files {
            let feature_name = get_gwas_feature_name(gwas_file);
            match self.feature_sets.gwas_files.insert(feature_name.clone()) {
                true => {
                    self.feature_sets
                        .gwas_feature_to_file
                        .insert(feature_name, gwas_file.clone());
                }
                false => bail!("GWAS file {} already exists", gwas_file.display()),
            }
        }
        Ok(())
    }

    pub fn register_feature_info(&mut self, feature_info_file: &FeatureInfoFile) -> Result<()> {
        let _span = info_span!("Registering feature info file").entered();
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
        // Load data
        let pheno_cols = &self.feature_sets.final_features;
        let mut data =
            read_phenotypes_covariates(pheno_cols, pheno_file_spec, covar_file_spec, true)?;

        // Compute sample sizes
        let dtypes: Vec<NodeType> = self
            .feature_sets
            .final_features
            .iter()
            .map(|x| self.feature_info_map.get(x).unwrap().type_.unwrap())
            .collect();
        let sample_sizes =
            compute_sample_size(&data.y_phenotypes, &dtypes, pheno_options.plink_offset)?;
        for (feature_code, sample_size) in self
            .feature_sets
            .final_features
            .iter()
            .zip(sample_sizes.iter())
        {
            self.feature_info_map
                .get_mut(feature_code)
                .unwrap()
                .sample_size = Some(*sample_size);
        }

        // Filter the data to remove features whose sample size is below the min sample size
        let keep_name_to_index = data
            .phenotype_names
            .iter()
            .enumerate()
            .filter(|(_, phenotype_name)| {
                let sample_size = self
                    .feature_info_map
                    .get(*phenotype_name)
                    .unwrap()
                    .sample_size
                    .unwrap();
                sample_size >= pheno_options.min_sample_size.unwrap_or(0) as f32
            })
            .map(|(index, phenotype_name)| (phenotype_name.clone(), index))
            .collect::<HashMap<String, usize>>();
        self.feature_sets
            .final_features
            .retain(|x| keep_name_to_index.contains_key(x));
        let mut new_y = Mat::zeros(
            data.y_phenotypes.nrows(),
            self.feature_sets.final_features.len(),
        );
        for (i, name) in self.feature_sets.final_features.iter().enumerate() {
            let index = keep_name_to_index.get(name).unwrap();
            new_y.col_mut(i).copy_from(&data.y_phenotypes.col(*index));
        }
        info!(
            "Filtered from {} to {} features based on sample size",
            data.y_phenotypes.ncols(),
            new_y.ncols()
        );
        data.y_phenotypes = new_y;
        data.phenotype_names = self.feature_sets.final_features.clone();

        let covariance_mat = {
            let _span = info_span!("Computing partial covariance matrix").entered();
            let y_resid = residualize_covariates(&data.x_covariates, &data.y_phenotypes)?;
            let ddof = self.metadata.num_covar.unwrap_or(0) as usize + 2;
            compute_covariance(&y_resid, ddof)
        };

        info!("Setting feature partial variances");
        for (i, feature_code) in self.feature_sets.final_features.iter().enumerate() {
            self.feature_info_map
                .get_mut(feature_code)
                .unwrap()
                .partial_variance = Some(covariance_mat[(i, i)]);
        }
        let mut covariance_df = {
            let _span = info_span!("Converting covariance matrix to dataframe").entered();
            mat_to_polars(covariance_mat, &data.phenotype_names)?
        };
        {
            let _span = info_span!("Writing covariance matrix").entered();
            let covar_path = self.cohort_directory.join("covariance.parquet");
            write_parquet(&mut covariance_df, &covar_path)?;
        }

        let anonymized_phenotypes = {
            let _span = info_span!("Anonymizing phenotypes").entered();
            let n_samples = match pheno_options.keep_n_samples {
                Some(n) => n,
                None => data.y_phenotypes.nrows(),
            };
            let raw_phenotypes = data
                .y_phenotypes
                .row_iter()
                .take(n_samples)
                .map(|x| x.iter().copied().collect::<Vec<f32>>())
                .collect::<Vec<Vec<f32>>>();
            mdav(raw_phenotypes, pheno_options.k_anonymity)?
        };
        let (anonymized_phenotypes_df, column_names) = {
            let _span = info_span!("Processing anonymized phenotypes").entered();
            let mut anonymized_phenotypes_df =
                vec_vec_to_polars(anonymized_phenotypes, &data.phenotype_names)?
                    .lazy()
                    .with_column(lit(1.0).cast(DataType::Float32).alias("intercept"))
                    .collect()?;
            let column_names = anonymized_phenotypes_df
                .get_column_names()
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<String>>();
            if pheno_options.plink_offset {
                let mut df = anonymized_phenotypes_df.lazy();
                for feature_code in self.feature_sets.final_features.iter() {
                    let dtype = self
                        .feature_info_map
                        .get(feature_code)
                        .expect("Failed to get feature info")
                        .type_
                        .expect("Feature type is missing");
                    if dtype == NodeType::Bool {
                        df = df.with_column(col(feature_code).sub(lit(2.0)))
                    }
                }
                anonymized_phenotypes_df = df.collect()?;
            }
            let pheno_path = self.cohort_directory.join("phenotypes.parquet");
            write_parquet(&mut anonymized_phenotypes_df, &pheno_path)?;
            (anonymized_phenotypes_df, column_names)
        };

        let left_inverse = {
            let _span = info_span!("Computing left inverse").entered();
            let anonymized_phenotype_mat = polars_to_faer_f32(anonymized_phenotypes_df.lazy())?;
            compute_left_inverse(&anonymized_phenotype_mat)?
                .transpose()
                .to_owned()
        };
        let _span = info_span!("Writing left inverse").entered();
        let mut left_inverse_df = mat_to_polars(left_inverse, &column_names)?;
        let left_inverse_path = self.cohort_directory.join("phenotype_left_inverse.parquet");
        write_parquet(&mut left_inverse_df, &left_inverse_path)?;
        Ok(())
    }

    pub fn process_gwas_files(
        &mut self,
        gwas_options: GWASOptions,
        variant_info_options: VariantInfoOptions,
    ) -> Result<()> {
        let variant_id = gwas_options.variant_id_column.clone();
        let a1 = gwas_options.a1_column.clone();
        let a2 = gwas_options.a2_column.clone();
        let beta = gwas_options.beta_column.clone();
        let std_err = gwas_options.std_err_column.clone();
        let sample_size = gwas_options.sample_size_column.clone();
        let column_specs = Arc::new([
            variant_id.clone().into(),
            a1.clone().into(),
            a2.clone().into(),
            beta.clone().into(),
            std_err.clone().into(),
            sample_size.clone().into(),
        ]);
        let schema_val: Vec<(PlSmallStr, DataType)> = vec![
            (variant_id.clone().into(), DataType::String),
            (a1.clone().into(), DataType::String),
            (a2.clone().into(), DataType::String),
            (beta.clone().into(), DataType::Float32),
            (std_err.clone().into(), DataType::Float32),
            (sample_size.clone().into(), DataType::Float32),
        ];
        let schema = Arc::new(Schema::from_iter(schema_val));
        let parse_opts = CsvParseOptions::default().with_separator(gwas_options.gwas_separator);
        let mut result_df = None;

        info!("Reading GWAS files");
        let pb = ProgressBar::new(self.feature_sets.final_features.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({eta})")
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
            .progress_chars("#>-")
        );
        for feature_name in self.feature_sets.final_features.iter() {
            let gwas_file = self
                .feature_sets
                .gwas_feature_to_file
                .get(feature_name)
                .expect("Failed to get GWAS file");
            let this_gwas_df = CsvReadOptions::default()
                .with_parse_options(parse_opts.clone())
                .with_columns(Some(column_specs.clone()))
                .with_schema_overwrite(Some(schema.clone()))
                .with_n_rows(gwas_options.keep_n_variants)
                .try_into_reader_with_file_path(Some(gwas_file.clone()))?
                .finish()
                .context(format!("Failed to read GWAS file {}", gwas_file.display()))?
                .lazy()
                .select([
                    col(&variant_id).alias("variant_id"),
                    col(&a1).alias("a1"),
                    col(&a2).alias("a2"),
                    col(&beta)
                        .cast(DataType::Float32)
                        .alias(format!("{}_beta", feature_name).as_str()),
                    col(&std_err)
                        .cast(DataType::Float32)
                        .alias(format!("{}_se", feature_name).as_str()),
                    col(&sample_size)
                        .cast(DataType::Int32)
                        .alias(format!("{}_ss", feature_name).as_str()),
                ])
                .collect()
                .context(format!("Failed to read GWAS file {}", gwas_file.display()))?;
            match result_df {
                None => {
                    result_df = Some(this_gwas_df);
                }
                Some(df) => {
                    let forward_df = df.inner_join(
                        &this_gwas_df,
                        ["variant_id", "a1", "a2"],
                        ["variant_id", "a1", "a2"],
                    )?;
                    let joined_df = if forward_df.height() == df.height() {
                        forward_df
                    } else {
                        let mut reverse_df = df.inner_join(
                            &this_gwas_df,
                            ["variant_id", "a1", "a2"],
                            ["variant_id", "a2", "a1"],
                        )?;
                        if reverse_df.height() > 0 {
                            reverse_df = reverse_df
                                .lazy()
                                .with_column(
                                    col(format!("{}_beta", feature_name).as_str()) * lit(-1.0_f32),
                                )
                                .collect()?;
                        }
                        forward_df.vstack(&reverse_df)?
                    };
                    result_df = Some(joined_df);
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("Done reading GWAS files");
        let result_df = result_df.expect("Failed to collect result df");
        info!("Processing loaded GWAS dataframe");
        let start = Instant::now();
        let mut lazy_result_df = result_df.lazy().select([
            col("variant_id"),
            col("a1"),
            col("a2"),
            min_horizontal([col("^*_ss$")])?
                .sub(lit(self.metadata.num_covar.unwrap_or(0) + 2))
                .alias("degrees_of_freedom"),
            col("^*_beta$"),
            col("^*_se$"),
        ]);
        for feature_name in self.feature_sets.final_features.iter() {
            let feature_variance = self
                .feature_info_map
                .get(feature_name)
                .expect("Failed to get feature variance");
            lazy_result_df = lazy_result_df.with_column(
                estimate_genotype_variance(
                    lit(feature_variance
                        .partial_variance
                        .expect("Failed to get partial variance")),
                    col("degrees_of_freedom"),
                    col(format!("{}_beta", feature_name).as_str()),
                    col(format!("{}_se", feature_name).as_str()),
                )
                .alias(format!("{}_gvar", feature_name).as_str()),
            );
        }
        let mut final_result_df = lazy_result_df
            .select([
                col("variant_id"),
                col("a1"),
                col("a2"),
                col("degrees_of_freedom"),
                mean_horizontal([col("^*_gvar$")])?
                    .cast(DataType::Float32)
                    .alias("genotype_partial_variance"),
                col("^*_beta$")
                    .name()
                    .map(|x| Ok(x.replace("_beta", "").into())),
            ])
            .collect()?;
        if variant_info_options.variant_info_columns.is_some() {
            info!("Adding variant info columns");
            let variant_info_df = self.process_variant_info_file(&variant_info_options)?;
            let mut column_order: Vec<String> =
                vec!["variant_id".to_string(), "a1".to_string(), "a2".to_string()];
            // Variant info columns are stored between a2 and degrees_of_freedom
            column_order.extend(variant_info_options.variant_info_columns.clone().unwrap());
            column_order.extend([
                "degrees_of_freedom".to_string(),
                "genotype_partial_variance".to_string(),
            ]);
            // GWAS columns follow genotype_partial_variance
            column_order.extend(
                final_result_df
                    .get_column_names()
                    .iter()
                    .skip(5)
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>(),
            );
            final_result_df = final_result_df
                .join(
                    &variant_info_df,
                    ["variant_id"],
                    ["variant_id"],
                    JoinArgs::new(JoinType::Inner),
                )?
                .select(column_order)?;
        }
        let duration = start.elapsed();
        info!("Processing loaded GWAS dataframe took {:?}", duration);
        let _span = info_span!("Writing GWAS dataframe").entered();
        let gwas_path = self.cohort_directory.join("gwas.parquet");
        write_parquet(&mut final_result_df, &gwas_path)?;
        Ok(())
    }

    pub fn process_feature_info_file(&mut self, feature_info_file: &FeatureInfoFile) -> Result<()> {
        let feature_info_df = {
            let _span = info_span!("Reading feature info file").entered();
            CsvReadOptions::default()
                .with_parse_options(
                    CsvParseOptions::default()
                        .with_separator(feature_info_file.feature_info_separator),
                )
                .with_columns(Some(Arc::new([
                    "code".into(),
                    "name".into(),
                    "type".into(),
                ])))
                .try_into_reader_with_file_path(Some(feature_info_file.feature_info_path.clone()))
                .context("Failed to set feature info file as path")?
                .finish()?
        };

        let _span = info_span!("Converting feature info file to rows").entered();
        self.feature_info_map = feature_info_df_to_vec(&feature_info_df)?
            .iter()
            .filter(|x| self.feature_sets.final_features.contains(&x.code))
            .map(|x| (x.code.clone(), x.clone()))
            .collect();
        Ok(())
    }

    pub fn process_variant_info_file(
        &mut self,
        variant_info_file_spec: &VariantInfoOptions,
    ) -> Result<DataFrame> {
        let _span = info_span!("Reading variant info file").entered();
        let mut cols = vec![variant_info_file_spec
            .variant_info_variant_id_column
            .clone()];
        cols.extend(
            variant_info_file_spec
                .variant_info_columns
                .clone()
                .unwrap_or_default(),
        );
        let mut variant_info_df = CsvReadOptions::default()
            .with_parse_options(
                CsvParseOptions::default()
                    .with_separator(variant_info_file_spec.variant_info_separator),
            )
            .try_into_reader_with_file_path(variant_info_file_spec.variant_info_path.clone())?
            .finish()
            .context("Failed to read variant info file")?
            .select(cols)?;
        variant_info_df.rename(
            &variant_info_file_spec.variant_info_variant_id_column,
            "variant_id".into(),
        )?;
        Ok(variant_info_df)
    }

    pub fn check_result(&self) -> Result<()> {
        let _span = info_span!("Checking result").entered();
        check_result(&self.cohort_directory)
    }

    pub fn cleanup(&mut self) -> Result<()> {
        match self.delete_cohort_from_database() {
            Ok(_) => {}
            Err(err) => {
                warn!("Failed to delete cohort from database: {}", err);
            }
        }
        std::fs::remove_dir_all(&self.cohort_directory)?;
        info!("Cleaned up all cohort data");
        Ok(())
    }
}

pub fn normalize_cohort_name(cohort_name: &str) -> String {
    cohort_name.to_lowercase().replace(" ", "_")
}

pub fn initialize_database(path: &Path) -> Result<rusqlite::Connection> {
    let db = rusqlite::Connection::open(path)?;
    db.execute_batch(
        "BEGIN;
        CREATE TABLE IF NOT EXISTS cohort (
            id INTEGER NOT NULL,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            num_covar INTEGER NOT NULL,
            PRIMARY KEY (id),
            UNIQUE (name),
            UNIQUE (normalized_name)
        );
        CREATE TABLE IF NOT EXISTS feature (
                id INTEGER NOT NULL,
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                cohort_id INTEGER,
                PRIMARY KEY (id),
                CONSTRAINT unique_feature UNIQUE (cohort_id, code, name),
                FOREIGN KEY(cohort_id) REFERENCES cohort (id)
        );
        COMMIT;",
    )?;
    Ok(db)
}

/// Check if the cohort already exists in the database
pub fn check_cohort_not_in_db(cohort_name: &str, db: &rusqlite::Connection) -> Result<()> {
    let mut stmt = db.prepare(
        "SELECT id, name, normalized_name, num_covar
            FROM cohort WHERE name = ?",
    )?;
    let existing_cohort = stmt
        .query_row([cohort_name], |row| {
            Ok(Cohort {
                id: row.get(0)?,
                name: row.get(1)?,
                normalized_name: row.get(2)?,
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

pub struct PhenotypesCovariates {
    pub y_phenotypes: Mat<f32>,
    pub x_covariates: Mat<f32>,
    pub phenotype_names: Vec<String>,
}

pub fn read_phenotypes_covariates(
    pheno_cols: &[String],
    pheno_file_spec: &PhenoFile,
    covar_file_spec: &CovarFile,
    add_intercept: bool,
) -> Result<PhenotypesCovariates> {
    let _span = info_span!("Reading phenotypes and covariates").entered();
    let mut select_cols = pheno_cols.iter().map(col).collect::<Vec<Expr>>();
    select_cols.push(col(&pheno_file_spec.pheno_person_id_column));
    let pheno_df = LazyCsvReader::new(&pheno_file_spec.pheno_path)
        .with_separator(pheno_file_spec.pheno_separator)
        .finish()
        .context("Failed to read phenotype file")?
        .select(&select_cols);
    let mut covar_df = LazyCsvReader::new(&covar_file_spec.covar_path)
        .with_separator(covar_file_spec.covar_separator)
        .finish()
        .context("Failed to read covariate file")?;
    let covar_ignore_cols = covar_file_spec
        .covar_ignore_columns
        .clone()
        .unwrap_or_default();
    covar_df = covar_df.drop(covar_ignore_cols);
    let covar_cols = covar_df
        .collect_schema()?
        .iter_names_cloned()
        .filter(|x| x != &covar_file_spec.covar_person_id_column)
        .collect::<Vec<PlSmallStr>>();
    let merged_df = pheno_df.inner_join(
        covar_df,
        col(&pheno_file_spec.pheno_person_id_column),
        col(&covar_file_spec.covar_person_id_column),
    );
    let y_df = merged_df.clone().collect()?.select(pheno_cols)?;
    assert_eq!(
        y_df.get_column_names()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>(),
        pheno_cols,
        "Phenotype columns are not in the correct order"
    );
    let y = polars_to_faer_f32(y_df.lazy()).context("Failed to convert phenotypes to faer")?;
    let mut x = merged_df.collect()?.select(covar_cols.clone())?;
    assert_eq!(
        x.get_column_names()
            .iter()
            .map(|x| x.to_string().into())
            .collect::<Vec<PlSmallStr>>(),
        covar_cols,
        "Covariate columns are not in the correct order"
    );
    if add_intercept {
        x.with_column(Column::new_scalar(
            "intercept".into(),
            Scalar::new(DataType::Float32, AnyValue::Float32(1.0)),
            x.height(),
        ))?;
    }
    let x = polars_to_faer_f32(x.lazy())?;
    info!(
        "Read phenotypes (shape {:?}) and covariates (shape {:?})",
        y.shape(),
        x.shape()
    );
    Ok(PhenotypesCovariates {
        y_phenotypes: y,
        x_covariates: x,
        phenotype_names: pheno_cols.to_vec(),
    })
}

pub fn mat_to_polars<'a, T>(mat: Mat<f32>, column_names: &'a [T]) -> Result<DataFrame>
where
    PlSmallStr: From<&'a T>,
{
    if mat.ncols() != column_names.len() {
        bail!("Matrix and column names must have the same length");
    }
    let columns = mat
        .col_iter()
        .zip(column_names)
        .map(|(x, y)| {
            Ok(Column::new(
                y.into(),
                x.iter().copied().collect::<Vec<f32>>(),
            ))
        })
        .collect::<Result<Vec<Column>>>()?;
    let df = DataFrame::new(columns)?;
    Ok(df)
}

/// Build a polars dataframe from a vector of vectors (rows)
pub fn vec_vec_to_polars(vecs: Vec<Vec<f32>>, column_names: &[String]) -> Result<DataFrame> {
    if vecs[0].len() != column_names.len() {
        bail!("Vectors and column names must have the same length");
    }
    let columns = transpose_vec_vec(vecs)
        .iter()
        .zip(column_names)
        .map(|(x, y)| Ok(Column::new(y.into(), x.to_vec())))
        .collect::<Result<Vec<Column>>>()?;
    let df = DataFrame::new(columns)?;
    Ok(df)
}

pub fn vec_vec_to_mat(vecs: Vec<Vec<f32>>) -> Mat<f32> {
    let mut mat = Mat::zeros(vecs.len(), vecs[0].len());
    for (i, row) in vecs.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            mat[(i, j)] = *x;
        }
    }
    mat
}

pub fn write_parquet(df: &mut DataFrame, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    ParquetWriter::new(writer).finish(df)?;
    Ok(())
}

pub fn estimate_genotype_variance(
    phenotype_variance: Expr,
    degrees_of_freedom: Expr,
    beta: Expr,
    std_error: Expr,
) -> Expr {
    phenotype_variance / (degrees_of_freedom * std_error.pow(2) + beta.pow(2))
}

pub fn compute_sample_size(
    data: &Mat<f32>,
    dtypes: &[NodeType],
    plink_offset: bool,
) -> Result<Vec<f32>> {
    let _span = info_span!("Computing sample sizes").entered();
    if data.ncols() != dtypes.len() {
        bail!("Data and dtypes must have the same length");
    }
    let mut sample_size = Vec::new();
    for (col, dtype) in data.col_iter().zip(dtypes.iter()) {
        let values = col.iter().copied();
        if dtype == &NodeType::Bool && plink_offset {
            let sum = values.map(|x| x - 2.0).sum::<f32>();
            sample_size.push(sum);
        } else if dtype == &NodeType::Bool {
            let sum = values.sum();
            sample_size.push(sum);
        } else if dtype == &NodeType::Real {
            let sum = values.filter(|x| !x.is_nan()).count() as f32;
            sample_size.push(sum);
        }
    }
    Ok(sample_size)
}

pub fn feature_info_df_to_vec(feature_info_df: &DataFrame) -> Result<Vec<FeatureInfo>> {
    izip!(
        feature_info_df.column("code")?.str()?.into_iter(),
        feature_info_df.column("name")?.str()?.into_iter(),
        feature_info_df.column("type")?.str()?.into_iter(),
    )
    .map(|(code, name, type_)| {
        Some(FeatureInfo {
            code: code?.to_string(),
            name: name?.to_string(),
            type_: Some(NodeType::from_str(type_?).unwrap()),
            sample_size: None,
            partial_variance: None,
        })
    })
    .collect::<Option<Vec<FeatureInfo>>>()
    .context("Failed to collect feature info rows")
}

pub struct FeatureInsertRow {
    pub code: String,
    pub name: String,
    pub type_: String,
    pub sample_size: f32,
}

pub fn find_differences(a: &[String], b: &[String]) -> Vec<String> {
    let left: Vec<String> = a.iter().filter(|x| !b.contains(x)).cloned().collect();
    let right: Vec<String> = b.iter().filter(|x| !a.contains(x)).cloned().collect();
    left.into_iter().chain(right).collect()
}

pub fn check_result(cohort_directory: &Path) -> Result<()> {
    let _span = info_span!("Checking result").entered();
    let cov_path = cohort_directory.join("covariance.parquet");
    let cov_file = File::open(cov_path)?;
    let cov_schema = ParquetReader::new(cov_file).schema()?;
    let cov_names = cov_schema
        .iter_names_cloned()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();

    let pheno_path = cohort_directory.join("phenotypes.parquet");
    let pheno_file = File::open(pheno_path)?;
    let pheno_schema = ParquetReader::new(pheno_file).schema()?;
    let mut pheno_names = pheno_schema
        .iter_names_cloned()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
    assert_eq!(
        pheno_names.last().unwrap(),
        "intercept",
        "Covariance and phenotype column names do not match"
    );
    pheno_names.pop();

    let li_path = cohort_directory.join("phenotype_left_inverse.parquet");
    let li_file = File::open(li_path)?;
    let li_schema = ParquetReader::new(li_file).schema()?;
    let mut li_names = li_schema
        .iter_names_cloned()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
    assert_eq!(
        li_names.last().unwrap(),
        "intercept",
        "Covariance and left inverse column names do not match"
    );
    li_names.pop();

    let gwas_path = cohort_directory.join("gwas.parquet");
    let gwas_file = File::open(gwas_path)?;
    let gwas_schema = ParquetReader::new(gwas_file).schema()?;
    let mut gwas_pheno_names = Vec::new();
    let mut past_metadata = false;
    for field in gwas_schema.iter_names() {
        if field == "genotype_partial_variance" {
            past_metadata = true;
            continue;
        }
        if past_metadata {
            gwas_pheno_names.push(field.to_string());
        }
    }

    assert_eq!(
        cov_names.len(),
        pheno_names.len(),
        "Covariance and phenotype column names do not match {:?}",
        find_differences(&cov_names, &pheno_names)
    );
    assert_eq!(
        cov_names.len(),
        li_names.len(),
        "Covariance and left inverse column names do not match {:?}",
        find_differences(&cov_names, &li_names)
    );
    assert_eq!(
        cov_names.len(),
        gwas_pheno_names.len(),
        "Covariance and GWAS column names do not match. Diff: {:?}",
        find_differences(&cov_names, &gwas_pheno_names)
    );

    izip!(
        cov_names.iter(),
        pheno_names.iter(),
        li_names.iter(),
        gwas_pheno_names.iter()
    )
    .for_each(|(cov_field, pheno_field, li_field, gwas_field)| {
        assert_eq!(
            cov_field, pheno_field,
            "Covariance and phenotype column names do not match"
        );
        assert_eq!(
            cov_field, li_field,
            "Covariance and left inverse column names do not match"
        );
        assert_eq!(
            cov_field, gwas_field,
            "Covariance and GWAS column names do not match"
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_compute_sample_size() {
        let data = mat![
            [1.0, 1.0, 1.0],
            [3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [7.0, 0.0, 0.0]
        ];
        let dtypes = vec![NodeType::Real, NodeType::Real, NodeType::Bool];
        let sample_size = compute_sample_size(&data, &dtypes, false).unwrap();
        assert_eq!(sample_size, vec![4.0, 4.0, 1.0]);
    }

    #[test]
    fn test_compute_sample_size_offset() {
        let data = mat![
            [1.0, 2.0, 3.0],
            [3.0, 3.0, 2.0],
            [5.0, 3.0, 2.0],
            [7.0, 3.0, 2.0]
        ];
        let dtypes = vec![NodeType::Real, NodeType::Real, NodeType::Bool];
        let sample_size = compute_sample_size(&data, &dtypes, true).unwrap();
        assert_eq!(sample_size, vec![4.0, 4.0, 1.0]);
    }
}
