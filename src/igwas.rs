use anyhow::{anyhow, Context, Result};
use faer::Col;
use faer_ext::polars::polars_to_faer_f32;
use itertools::izip;
use log::debug;
use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::{fs::File, path::Path};

use crate::utils::{slice_after_excl, slice_before, slice_before_excl};

#[derive(Debug)]
pub struct Projection {
    pub feature_id: Vec<String>,
    pub feature_coefficient: Col<f32>,
    pub n_features: usize,
}

impl Projection {
    pub fn new(feature_id: Vec<String>, feature_coefficient: Col<f32>) -> Result<Self> {
        if feature_id.len() != feature_coefficient.nrows() {
            return Err(anyhow!(
                "feature_id and feature_coefficient must have the same length.",
            ));
        }
        let mut final_feature_id = Vec::new();
        let mut final_feature_coefficient = Vec::new();
        for (feature_id, feature_coefficient) in feature_id.iter().zip(feature_coefficient.iter()) {
            if feature_coefficient != &0.0 {
                final_feature_id.push(feature_id.clone());
                final_feature_coefficient.push(*feature_coefficient);
            }
        }
        assert_eq!(final_feature_coefficient.len(), final_feature_id.len());
        let n_features = final_feature_coefficient.len();
        let mut feature_coef_col = Col::zeros(n_features);
        final_feature_coefficient
            .iter()
            .enumerate()
            .for_each(|(i, x)| {
                feature_coef_col[i] = *x;
            });
        let result = Self {
            feature_id: final_feature_id,
            feature_coefficient: feature_coef_col,
            n_features,
        };
        Ok(result)
    }

    pub fn remove_zeros(&mut self) {
        let mut n_features = 0;
        let mut new_feature_id = Vec::new();
        let mut new_feature_coefficient = Vec::new();
        for (feature_id, feature_coefficient) in
            self.feature_id.iter().zip(self.feature_coefficient.iter())
        {
            if feature_coefficient != &0.0 {
                new_feature_id.push(feature_id.clone());
                new_feature_coefficient.push(*feature_coefficient);
                n_features += 1;
            }
        }
        assert_eq!(n_features, new_feature_coefficient.len());
        assert_eq!(n_features, new_feature_id.len());
        self.n_features = n_features;
        self.feature_id = new_feature_id;
        let mut new_feature_coefficient_col = Col::zeros(n_features);
        new_feature_coefficient
            .iter()
            .enumerate()
            .for_each(|(i, x)| {
                new_feature_coefficient_col[i] = *x;
            });
        self.feature_coefficient = new_feature_coefficient_col;
    }

    pub fn standardize(&mut self, full_feature_ids: &[String]) {
        assert_eq!(self.n_features, self.feature_coefficient.nrows());
        assert_eq!(self.n_features, self.feature_id.len());
        let mut new_feature_coefficient = Vec::new();
        for feature_id in full_feature_ids {
            match self.feature_id.contains(feature_id) {
                true => {
                    let original_index = self
                        .feature_id
                        .iter()
                        .position(|x| x == feature_id)
                        .unwrap();
                    new_feature_coefficient.push(self.feature_coefficient[original_index]);
                }
                false => {
                    new_feature_coefficient.push(0.0);
                }
            }
        }
        let mut new_feature_coefficient_col = Col::zeros(full_feature_ids.len());
        new_feature_coefficient
            .iter()
            .enumerate()
            .for_each(|(i, x)| {
                new_feature_coefficient_col[i] = *x;
            });
        self.feature_coefficient = new_feature_coefficient_col;
        self.feature_id = full_feature_ids.to_vec();
        self.n_features = full_feature_ids.len();
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &f32)> {
        self.feature_id.iter().zip(self.feature_coefficient.iter())
    }
}

pub fn indirect_std_error(
    projection_variance: f32,
    indirect_genotype_variance: f32,
    indirect_beta: f32,
    indirect_degrees_of_freedom: i32,
) -> f32 {
    let variance_ratio = projection_variance / indirect_genotype_variance;
    let numerator = variance_ratio - indirect_beta.powi(2);
    (numerator / indirect_degrees_of_freedom as f32).sqrt()
}

#[derive(Clone, Debug)]
pub struct SingleVariantRunningStats {
    pub variant_id: String,
    pub beta: f32,
    pub genotype_variance: f32,
    pub degrees_of_freedom: i32,
}

pub struct RunningStats {
    pub variant_id: Vec<String>,
    pub a1: Vec<String>,
    pub a2: Vec<String>,
    pub info: Vec<Column>,
    pub beta: Vec<f32>,
    pub genotype_variance: Vec<f32>,
    pub degrees_of_freedom: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct SingleVariantResultStats {
    pub variant_id: String,
    pub beta: f32,
    pub std_error: f32,
    pub t_stat: f32,
    pub neg_log_p_value: f32,
    pub sample_size: i32,
}

#[derive(Clone, Debug)]
pub struct ResultStats {
    pub variant_id: Vec<String>,
    pub a1: Vec<String>,
    pub a2: Vec<String>,
    pub info: Vec<Column>,
    pub beta: Vec<f32>,
    pub std_error: Vec<f32>,
    pub t_stat: Vec<f32>,
    pub neg_log_p_value: Vec<f32>,
    pub sample_size: Vec<i32>,
}

pub struct FeatureStats {
    pub variant_id: Option<Vec<String>>,
    pub beta: Float32Chunked,
    pub genotype_variance: Float32Chunked,
    pub degrees_of_freedom: Int32Chunked,
}

pub fn compute_batch_stats(df: &DataFrame, projection: &mut Projection) -> Result<RunningStats> {
    let columns = df
        .get_column_names()
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
    let cols_to_drop = slice_before(&columns, &"genotype_partial_variance".to_string());
    let gwas_beta_df = df.drop_many(&cols_to_drop);
    let feature_ids = gwas_beta_df
        .get_column_names()
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
    let gwas_beta_mat = polars_to_faer_f32(gwas_beta_df.lazy())?;
    projection.standardize(&feature_ids);
    assert_eq!(
        projection.n_features,
        projection.feature_coefficient.nrows()
    );
    let gwas_beta = (gwas_beta_mat * &projection.feature_coefficient)
        .iter()
        .cloned()
        .collect::<Vec<f32>>();
    let variant_id = df
        .column("variant_id")
        .context("No variant_id column")?
        .str()
        .context("variant_id column is not str")?
        .iter()
        .map(|x| x.expect("Failed to get variant_id").to_string())
        .collect::<Vec<String>>();
    let a1 = df
        .column("a1")
        .context("No a1 column")?
        .str()
        .context("a1 column is not str")?
        .iter()
        .map(|x| x.expect("Failed to get a1").to_string())
        .collect::<Vec<String>>();
    let a2 = df
        .column("a2")
        .context("No a2 column")?
        .str()
        .context("a2 column is not str")?
        .iter()
        .map(|x| x.expect("Failed to get a2").to_string())
        .collect::<Vec<String>>();
    let genotype_variance = df
        .column("genotype_partial_variance")
        .context("No genotype_variance column")?
        .f32()
        .context("genotype_partial_variance column is not f32")?
        .iter()
        .collect::<Option<Vec<f32>>>()
        .expect("Failed to collect genotype_partial_variance");
    let degrees_of_freedom = df
        .column("degrees_of_freedom")
        .context("No degrees_of_freedom column")?
        .i32()
        .context("Degrees_of_freedom column is not i32")?
        .iter()
        .collect::<Option<Vec<i32>>>()
        .expect("Failed to collect degrees_of_freedom");
    let info_cols = slice_after_excl(&cols_to_drop, &"a2".to_string());
    let info_cols = slice_before_excl(&info_cols, &"degrees_of_freedom".to_string());
    let info = df.select_columns(info_cols)?;
    Ok(RunningStats {
        variant_id,
        a1,
        a2,
        info,
        beta: gwas_beta,
        genotype_variance,
        degrees_of_freedom,
    })
}

pub fn compute_batch_results(
    running_stats: RunningStats,
    projection_variance: f32,
    n_covariates: usize,
) -> Result<ResultStats> {
    let std_error: Vec<f32> = izip!(
        &running_stats.beta,
        &running_stats.genotype_variance,
        &running_stats.degrees_of_freedom
    )
    .map(|(beta, genotype_variance, degrees_of_freedom)| {
        indirect_std_error(
            projection_variance,
            *genotype_variance,
            *beta,
            *degrees_of_freedom,
        )
    })
    .collect();

    let t_stat: Vec<f32> = running_stats
        .beta
        .iter()
        .zip(std_error.iter())
        .map(|(beta, std_error)| beta / std_error)
        .collect();

    let neg_log_p_value: Vec<f32> = t_stat
        .iter()
        .zip(running_stats.degrees_of_freedom.iter())
        .map(|(&t_stat, &dof)| compute_neg_log_pvalue(t_stat, dof))
        .collect();

    let sample_size: Vec<i32> = running_stats
        .degrees_of_freedom
        .iter()
        .map(|dof| dof + n_covariates as i32 + 2)
        .collect();

    Ok(ResultStats {
        variant_id: running_stats.variant_id,
        a1: running_stats.a1,
        a2: running_stats.a2,
        info: running_stats.info,
        beta: running_stats.beta,
        std_error,
        t_stat,
        neg_log_p_value,
        sample_size,
    })
}

pub fn results_to_dataframe(result_stats: ResultStats) -> Result<DataFrame> {
    let mut cols = vec![
        Column::new("variant_id".into(), result_stats.variant_id),
        Column::new("a1".into(), result_stats.a1),
        Column::new("a2".into(), result_stats.a2),
    ];
    cols.extend(result_stats.info);
    cols.extend([
        Column::new("beta".into(), result_stats.beta),
        Column::new("std_error".into(), result_stats.std_error),
        Column::new("t_stat".into(), result_stats.t_stat),
        Column::new("neg_log_p_value".into(), result_stats.neg_log_p_value),
        Column::new("sample_size".into(), result_stats.sample_size),
    ]);
    let df = DataFrame::new(cols)?;
    Ok(df)
}

pub fn write_dataframe(
    df: &mut DataFrame,
    path: &Path,
    n_threads: usize,
    compress: bool,
) -> Result<()> {
    let file = File::create(path)?;
    if compress {
        let compressed_writer = zstd::Encoder::new(file, 3)?.auto_finish();
        CsvWriter::new(compressed_writer)
            .with_separator(b'\t')
            .n_threads(n_threads)
            .finish(df)?;
    } else {
        CsvWriter::new(file)
            .with_separator(b'\t')
            .n_threads(n_threads)
            .finish(df)?;
    }
    Ok(())
}

pub fn run_igwas_df_impl(
    gwas_df: &DataFrame,
    projection: &mut Projection,
    projection_variance: f32,
    n_covariates: usize,
    output_path: &Path,
    n_threads: usize,
) -> Result<()> {
    debug!("Computing batch stats");
    let running_stats = compute_batch_stats(gwas_df, projection)?;
    debug!("Computing batch results");
    let result_stats = compute_batch_results(running_stats, projection_variance, n_covariates)?;
    debug!("Converting results to dataframe");
    let mut results_df = results_to_dataframe(result_stats)?;
    debug!("Writing results");
    write_dataframe(&mut results_df, output_path, n_threads, false)?;
    Ok(())
}

pub fn compute_neg_log_pvalue(t_statistic: f32, degrees_of_freedom: i32) -> f32 {
    let t = t_statistic as f64;
    let dof = degrees_of_freedom as f64;

    match t {
        f if f.is_nan() => f32::NAN,
        f if f.is_infinite() => f32::INFINITY,
        _ => {
            if dof <= 1.0 {
                return f32::NAN;
            }
            let t_dist = StudentsT::new(0.0, 1.0, dof)
                .with_context(|| format!("Failed to compute t-statistic for dof {}", dof))
                .unwrap();
            let p = 2.0 * t_dist.cdf(-t.abs());
            -p.log10() as f32
        }
    }
}
