use axum::{
    extract::{Path, State},
    Json,
};
use itertools::{izip, Itertools};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, Pool, Sqlite};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
};
use thiserror::Error;
use uuid::Uuid;

use crate::{models::round_to_decimals, models::WebGWASResultStatus, AppState};

#[derive(Deserialize)]
pub struct PvaluesQuery {
    pub cohort_id: Option<i32>,
    pub features: Option<Vec<String>>,
    #[serde(rename = "minp")]
    pub min_neg_log_p: Option<f32>,
}

#[derive(Serialize)]
pub struct PvaluesResponse {
    pub request_id: Uuid,
    pub status: WebGWASResultStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_msg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pvalues: Option<Vec<Pvalue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chromosome_positions: Option<Vec<ChromosomePosition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_map: Option<HashMap<i32, String>>,
}

#[derive(Debug, Serialize)]
pub struct Pvalue {
    #[serde(rename = "i")]
    pub index: i32,
    #[serde(rename = "p", serialize_with = "round_to_decimals")]
    pub pvalue: f32,
    #[serde(rename = "c")]
    pub chromosome: String,
    #[serde(rename = "g")]
    pub color: i32,
    #[serde(rename = "l")]
    pub label: String,
    #[serde(skip_serializing)]
    pub chrbp: String,
}

#[derive(Serialize)]
pub struct ChromosomePosition {
    #[serde(rename = "c")]
    pub chromosome: String,
    #[serde(rename = "m")]
    pub midpoint: i32,
}

#[axum::debug_handler]
pub async fn get_igwas_pvalues(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<Uuid>,
    Json(request): Json<PvaluesQuery>,
) -> Json<PvaluesResponse> {
    match process_request(&state, &request_id, &request).await {
        Ok(response) => response,
        Err(err) => err.to_json_response(&request_id),
    }
}

async fn process_request(
    state: &AppState,
    request_id: &Uuid,
    request: &PvaluesQuery,
) -> Result<Json<PvaluesResponse>, AppError> {
    let results = state
        .results
        .lock()
        .unwrap()
        .get(request_id)
        .cloned()
        .ok_or_else(|| AppError::NoResult(request_id.to_string()))?;
    let results_path = results
        .local_result_file
        .as_ref()
        .ok_or_else(|| AppError::NoLocalFile(request_id.to_string()))?;

    let mut result = load_pvalues(results_path.to_path_buf(), request.min_neg_log_p)?;

    if let Some(features) = &request.features {
        let cohort_id = request.cohort_id.ok_or(AppError::CohortIdMissing)?;
        validate_features(state, cohort_id, features)?;

        let hits = get_hits(&state.db, features, cohort_id)
            .await
            .map_err(|_| AppError::HitQueryFailed)?;

        result = color_hits(result, hits);
    }

    Ok(Json(PvaluesResponse {
        request_id: *request_id,
        status: WebGWASResultStatus::Done,
        error_msg: None,
        pvalues: Some(result.pvalues),
        chromosome_positions: Some(result.chromosome_positions),
        color_map: Some(result.color_map),
    }))
}

#[derive(Error, Debug)]
enum AppError {
    #[error("No result found for request: {0}")]
    NoResult(String),
    #[error("No local file found for request: {0}")]
    NoLocalFile(String),
    #[error("Failed to load p-values: {0}")]
    PValueLoadFailed(#[from] PolarsError),
    #[error("Error processing: {0}")]
    ProcessingError(String),
    #[error("Bad query: feature `{0}` not found")]
    FeatureNotFound(String),
    #[error("Bad query: cohort_id `{0}` not found")]
    CohortNotFound(String),
    #[error("Bad query: cohort_id not given")]
    CohortIdMissing,
    #[error("Failed to query hits")]
    HitQueryFailed,
}

impl AppError {
    fn to_json_response(&self, request_id: &Uuid) -> Json<PvaluesResponse> {
        Json(PvaluesResponse {
            request_id: *request_id,
            status: WebGWASResultStatus::Error,
            error_msg: Some(self.to_string()),
            pvalues: None,
            chromosome_positions: None,
            color_map: None,
        })
    }
}

fn validate_features(
    state: &AppState,
    cohort_id: i32,
    features: &[String],
) -> Result<(), AppError> {
    let cohort_data = {
        let binding = state.cohort_id_to_data.lock().unwrap();
        binding
            .get(&cohort_id)
            .cloned()
            .ok_or_else(|| AppError::CohortNotFound(cohort_id.to_string()))?
    };

    for feature in features {
        if !cohort_data.feature_names.contains(feature) {
            return Err(AppError::FeatureNotFound(feature.clone()));
        }
    }

    Ok(())
}

struct PvaluesResult {
    pvalues: Vec<Pvalue>,
    chromosome_positions: Vec<ChromosomePosition>,
    color_map: HashMap<i32, String>,
}

/// Load p-values from a result file
fn load_pvalues(path: PathBuf, min_neg_log_p: Option<f32>) -> Result<PvaluesResult, AppError> {
    let mut df = read_pvalue_df(path)?;
    if let Some(min_neg_log_p) = min_neg_log_p {
        df = df
            .lazy()
            .filter(col("neg_log_p_value").gt_eq(min_neg_log_p))
            .collect()
            .map_err(|_| AppError::ProcessingError("Error filtering p-values".to_string()))?;
    }
    let index_col = build_chromosome_index(&df)?;
    let df = df
        .with_column(index_col)
        .map_err(|_| AppError::ProcessingError("Error adding index column".to_string()))?
        .sort(
            ["chromosome_index".to_string(), "position".to_string()],
            SortMultipleOptions::default(),
        )
        .map_err(|_| AppError::ProcessingError("Error sorting dataframe".to_string()))?;
    let pvalues = extract_pvalue_vec(&df)?;
    let chromosome_positions = extract_chromosome_positions(&pvalues);
    let color_map = (0..23)
        .map(|x| (x, index_to_chrom(x)))
        .collect::<HashMap<i32, String>>();
    Ok(PvaluesResult {
        pvalues,
        chromosome_positions,
        color_map,
    })
}

fn read_pvalue_df(path: PathBuf) -> Result<DataFrame, AppError> {
    let schema_override = Schema::from_iter(vec![
        ("neg_log_p_value".into(), DataType::Float32),
        ("chromosome".into(), DataType::String),
        ("position".into(), DataType::Int32),
    ]);
    let df = CsvReadOptions::default()
        .with_schema_overwrite(Some(Arc::new(schema_override)))
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .try_into_reader_with_file_path(Some(path))?
        .finish()?;
    Ok(df)
}

fn chrom_to_index(chrom: &str) -> i32 {
    match chrom {
        "X" => 22,
        "Y" => 23,
        _ => chrom.parse::<i32>().expect("Failed to parse chromosome") - 1,
    }
}

fn index_to_chrom(index: i32) -> String {
    match index {
        22 => "X".to_string(),
        23 => "Y".to_string(),
        _ => format!("{}", index + 1),
    }
}

/// Create a column that indexes the chromosomes in order (1, 2, ..., 22, X, Y)
fn build_chromosome_index(df: &DataFrame) -> Result<Column, AppError> {
    let mut result: Column = df
        .column("chromosome")
        .map_err(|_| AppError::ProcessingError("`chromosome` column not found".to_string()))?
        .str()
        .map_err(|_| AppError::ProcessingError("`chromosome` column is not a string".to_string()))?
        .clone()
        .iter()
        .map(|chr_opt| chr_opt.map(chrom_to_index))
        .collect::<Int32Chunked>()
        .into_column();
    result.rename("chromosome_index".into());
    Ok(result)
}

fn extract_pvalue_vec(df: &DataFrame) -> Result<Vec<Pvalue>, AppError> {
    let pvalues: Vec<Pvalue> = izip!(
        df.column("variant_id")
            .map_err(|_| AppError::ProcessingError("`variant_id` column not found".to_string()))?
            .str()
            .map_err(|_| AppError::ProcessingError(
                "`variant_id` column is not string".to_string()
            ))?
            .into_iter(),
        df.column("neg_log_p_value")
            .map_err(|_| AppError::ProcessingError(
                "`neg_log_p_value` column not found".to_string()
            ))?
            .f32()
            .map_err(|_| AppError::ProcessingError(
                "`neg_log_p_value` column is not f32".to_string()
            ))?
            .into_iter(),
        df.column("chromosome")
            .map_err(|_| AppError::ProcessingError("`chromosome` column not found".to_string()))?
            .str()
            .map_err(|_| AppError::ProcessingError(
                "`chromosome` column is not string".to_string()
            ))?
            .into_iter(),
        df.column("chromosome_index")
            .map_err(|_| AppError::ProcessingError(
                "`chromosome_index` column not found".to_string()
            ))?
            .i32()
            .map_err(|_| AppError::ProcessingError(
                "`chromosome_index` column is not i32".to_string()
            ))?
            .into_iter(),
        df.column("rsid")
            .map_err(|_| AppError::ProcessingError("`rsid` column not found".to_string()))?
            .str()
            .map_err(|_| AppError::ProcessingError("`rsid` column is not string".to_string()))?
            .into_iter(),
    )
    .enumerate()
    .map(|(i, (chrbp, pvalue, chromosome, chr_idx, rsid))| Pvalue {
        index: i as i32,
        pvalue: pvalue.expect("Failed to get pvalue"),
        chromosome: chromosome.expect("Failed to get chromosome").to_string(),
        color: chr_idx.expect("Failed to get chromosome"),
        label: rsid.expect("Failed to get rsid").to_string(),
        chrbp: chrbp.expect("Failed to get chrbp").to_string(),
    })
    .collect();
    Ok(pvalues)
}

#[derive(Copy, Clone, Debug)]
struct ChromosomeBounds {
    min: i32,
    max: i32,
}

fn extract_chromosome_positions(pvalues: &[Pvalue]) -> Vec<ChromosomePosition> {
    let mut chromosomes = [ChromosomeBounds {
        min: i32::MAX,
        max: 0,
    }; 24];
    pvalues.iter().for_each(|x| {
        let chrom_idx = chrom_to_index(&x.chromosome) as usize;
        if x.index < chromosomes[chrom_idx].min {
            chromosomes[chrom_idx].min = x.index;
        }
        if x.index > chromosomes[chrom_idx].max {
            chromosomes[chrom_idx].max = x.index;
        }
    });
    let result = chromosomes
        .iter()
        .enumerate()
        .filter(|(_, x)| x.min < x.max) // Only keep those that have been initialized
        .map(|(i, x)| ChromosomePosition {
            chromosome: index_to_chrom(i as i32),
            midpoint: (x.min + x.max) / 2,
        })
        .collect();
    result
}

#[derive(FromRow)]
struct SignificantHit {
    feature_code: String,
    variant_id: String,
}

async fn get_hits(
    db: &Pool<Sqlite>,
    features: &[String],
    cohort_id: i32,
) -> Result<Vec<SignificantHit>, sqlx::Error> {
    let csv = features
        .iter()
        .map(|x| format!("'{}'", x))
        .collect::<Vec<String>>()
        .join(", ");
    let query_string = format!(
        "SELECT feature_code, variant_id
        FROM hits
        WHERE cohort_id = $1 AND feature_code IN ({})
        ORDER BY variant_id, feature_code",
        csv
    );
    let result = sqlx::query_as::<_, SignificantHit>(&query_string)
        .bind(cohort_id)
        .fetch_all(db)
        .await?;
    Ok(result)
}

fn color_hits(input: PvaluesResult, hits: Vec<SignificantHit>) -> PvaluesResult {
    let mut variant_to_features: HashMap<String, Vec<String>> = HashMap::new();
    for hit in hits {
        variant_to_features
            .entry(hit.variant_id)
            .or_default()
            .push(hit.feature_code);
    }
    let variant_to_hit_combo = variant_to_features
        .into_iter()
        .map(|(variant_id, mut combo)| {
            combo.sort(); // Most important part!
            let combo_string = combo.join(", ");
            (variant_id, combo_string)
        })
        .collect::<HashMap<String, String>>();
    let unique_combos = variant_to_hit_combo
        .values()
        .unique()
        .cloned()
        .collect::<HashSet<String>>();
    let mut max_current = *input.color_map.keys().max().unwrap_or(&23);
    let mut color_map = input.color_map;
    let mut combo_to_color = HashMap::new();
    for combo in unique_combos.into_iter() {
        max_current += 1;
        color_map.insert(max_current, combo.clone());
        combo_to_color.insert(combo, max_current);
    }
    let pvalues = input
        .pvalues
        .into_iter()
        .map(|x| {
            let color = match variant_to_hit_combo.get(&x.chrbp) {
                Some(combo) => *combo_to_color.get(combo).unwrap(),
                None => x.color,
            };
            Pvalue {
                index: x.index,
                pvalue: x.pvalue,
                chromosome: x.chromosome,
                color,
                label: x.label,
                chrbp: x.chrbp,
            }
        })
        .collect::<Vec<Pvalue>>();
    PvaluesResult {
        pvalues,
        chromosome_positions: input.chromosome_positions,
        color_map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrom_to_index() {
        assert_eq!(chrom_to_index("X"), 22);
        assert_eq!(chrom_to_index("Y"), 23);
        assert_eq!(chrom_to_index("1"), 0);
        assert_eq!(chrom_to_index("2"), 1);
        assert_eq!(chrom_to_index("3"), 2);
    }

    #[test]
    fn test_extract_chromosome_positions() {
        let pvalues = vec![
            Pvalue {
                index: 0,
                pvalue: 0.1,
                chromosome: "1".to_string(),
                color: 1,
                label: "rs1".to_string(),
                chrbp: "1:1".to_string(),
            },
            Pvalue {
                index: 1,
                pvalue: 0.2,
                chromosome: "1".to_string(),
                color: 1,
                label: "rs2".to_string(),
                chrbp: "1:2".to_string(),
            },
            Pvalue {
                index: 2,
                pvalue: 0.3,
                chromosome: "Y".to_string(),
                color: 2,
                label: "rs3".to_string(),
                chrbp: "Y:1".to_string(),
            },
            Pvalue {
                index: 3,
                pvalue: 0.4,
                chromosome: "Y".to_string(),
                color: 2,
                label: "rs4".to_string(),
                chrbp: "Y:2".to_string(),
            },
        ];
        let result = extract_chromosome_positions(&pvalues);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chromosome, "1");
        assert_eq!(result[1].chromosome, "Y");
    }
}
