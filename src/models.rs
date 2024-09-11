use anyhow::{anyhow, Result};
use log::debug;
use ndarray::Array2;
use polars::{
    datatypes::Float32Type,
    io::{csv::read::CsvReadOptions, parquet::read::ParquetReader, SerReader},
    prelude::{DataFrame, IndexOrder},
};
use serde::{Deserialize, Serialize};
use sqlx::prelude::FromRow;
use std::fs::{read_to_string, File};
use std::str::FromStr;
use std::{fmt::Display, path::Path};
use tokio::time::Instant;
use uuid::Uuid;

#[derive(Serialize)]
pub struct CohortResponse {
    pub id: i32,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, FromRow, Serialize)]
pub struct Cohort {
    pub id: i32,
    pub name: String,
    pub root_directory: String,
    pub num_covar: i32,
}

#[derive(Serialize, PartialEq, Clone, Copy, Debug, Deserialize, sqlx::Type)]
#[sqlx(rename_all = "UPPERCASE")]
pub enum NodeType {
    #[serde(rename = "BOOL")]
    Bool,
    #[serde(rename = "REAL")]
    Real,
    #[serde(rename = "ANY")]
    Any,
}

impl Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Serialize)]
pub struct FeatureResponse {
    pub id: i32,
    pub code: String,
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: NodeType,
}

#[derive(Clone, Debug, PartialEq, FromRow, Serialize)]
pub struct Feature {
    pub id: i32,
    pub code: String,
    pub name: String,
    pub node_type: NodeType,
    pub cohort_id: i32,
}

#[derive(Serialize)]
pub struct ValidPhenotypeResponse {
    pub is_valid: bool,
    pub message: String,
    pub phenotype_definition: String,
}

pub struct ValidPhenotype {
    pub is_valid: bool,
    pub message: String,
    pub phenotype_definition: String,
    pub valid_nodes: Vec<Node>,
}

#[derive(Deserialize, sqlx::Type)]
pub struct WebGWASRequest {
    pub phenotype_definition: String,
    pub cohort_id: i32,
}

pub struct WebGWASRequestId {
    pub id: Uuid,
    pub request_time: Instant,
    pub phenotype_definition: Vec<Node>,
    pub cohort_id: i32,
}

#[derive(Clone, Serialize)]
pub enum WebGWASResultStatus {
    Queued,
    Done,
    Error,
}

#[derive(Serialize)]
pub struct WebGWASResponse {
    pub request_id: Uuid,
    pub status: WebGWASResultStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Serialize)]
pub struct WebGWASResult {
    pub request_id: Uuid,
    pub status: WebGWASResultStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_msg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Serialize)]
pub struct ApproximatePhenotypeValues {
    pub true_value: f32,
    pub approx_value: f32,
    pub n: i32,
}

#[derive(Clone, Serialize)]
pub struct PhenotypeFitQuality {
    pub phenotype_fit_quality: f32,
    pub gwas_fit_quality: f32,
}

#[derive(Serialize)]
pub struct PhenotypeSummary {
    pub phenotype_definition: String,
    pub cohort_id: i32,
    pub phenotype_values: Vec<ApproximatePhenotypeValues>,
    pub fit_quality_reference: Vec<PhenotypeFitQuality>,
    pub rsquared: f32,
}

#[derive(Clone, Debug)]
pub struct Operator {
    pub id: i32,
    pub name: String,
    pub arity: i32,
    pub input_type: NodeType,
    pub output_type: NodeType,
}

#[derive(Clone, Debug)]
pub struct Constant {
    pub value: f32,
    pub node_type: NodeType,
}

#[derive(Debug, Clone)]
pub enum Node {
    Feature(Feature),
    Operator(Operators),
    Constant(Constant),
}

#[derive(Deserialize)]
pub struct GetFeaturesRequest {
    pub cohort_id: i32,
}

pub struct CohortInfo {
    pub cohort_id: i32,
    pub cohort_name: String,
    pub num_covar: i32,
    pub features_df: DataFrame,
    pub left_inverse: Array2<f32>,
    pub gwas_df: DataFrame,
    pub covariance_matrix: Array2<f32>,
}

#[derive(Deserialize)]
pub struct CohortMetadata {
    pub cohort_id: i32,
    pub cohort_name: String,
    pub num_covar: i32,
}

impl CohortInfo {
    pub fn load(root_directory: &Path) -> Result<CohortInfo> {
        debug!("Loading metadata for {}", root_directory.display());
        let metadata_file_path = root_directory.join("metadata.toml");
        let metadata = read_to_string(metadata_file_path)?;
        let metadata = toml::from_str::<CohortMetadata>(&metadata)?;

        debug!("Loading features for {}", root_directory.display());
        let features_file_path = root_directory.join("phenotype_data.parquet");
        let features_file = File::open(features_file_path)?;
        let features_df = ParquetReader::new(features_file).finish()?;

        debug!("Loading left inverse for {}", root_directory.display());
        let left_inverse_file_path = root_directory.join("phenotype_left_inverse.parquet");
        let left_inverse_file = File::open(left_inverse_file_path)?;
        let mut left_inverse = ParquetReader::new(left_inverse_file)
            .finish()?
            .to_ndarray::<Float32Type>(IndexOrder::Fortran)?;
        left_inverse.swap_axes(0, 1); // Transpose

        debug!("Loading GWAS for {}", root_directory.display());
        let gwas_file_path = root_directory.join("gwas.parquet");
        let gwas_file = File::open(gwas_file_path)?;
        let gwas_df = ParquetReader::new(gwas_file).finish()?;

        debug!("Loading covariance matrix for {}", root_directory.display());
        let covariance_matrix_file_path = root_directory.join("phenotypic_covariance.csv");
        let covariance_matrix = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(covariance_matrix_file_path))?
            .finish()?
            .drop("phenotype")?
            .to_ndarray::<Float32Type>(IndexOrder::Fortran)?;

        debug!(
            "Finished loading cohort info for {}",
            root_directory.display()
        );

        Ok(CohortInfo {
            cohort_id: metadata.cohort_id,
            cohort_name: metadata.cohort_name,
            num_covar: metadata.num_covar,
            features_df,
            left_inverse,
            gwas_df,
            covariance_matrix,
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Operators {
    Root,
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Not,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

impl FromStr for Operators {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "ROOT" => Ok(Operators::Root),
            "ADD" => Ok(Operators::Add),
            "SUB" => Ok(Operators::Sub),
            "MUL" => Ok(Operators::Mul),
            "DIV" => Ok(Operators::Div),
            "AND" => Ok(Operators::And),
            "OR" => Ok(Operators::Or),
            "NOT" => Ok(Operators::Not),
            "GT" => Ok(Operators::Gt),
            "GE" => Ok(Operators::Ge),
            "LT" => Ok(Operators::Lt),
            "LE" => Ok(Operators::Le),
            "EQ" => Ok(Operators::Eq),
            _ => Err(anyhow!("Invalid operator: {}", s)),
        }
    }
}

impl Operators {
    pub fn value(&self) -> Operator {
        match self {
            Operators::Root => Operator {
                id: 0,
                name: "root".to_string(),
                arity: 1,
                input_type: NodeType::Any,
                output_type: NodeType::Any,
            },
            Operators::Add => Operator {
                id: 1,
                name: "add".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Real,
            },
            Operators::Sub => Operator {
                id: 2,
                name: "sub".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Real,
            },
            Operators::Mul => Operator {
                id: 3,
                name: "mul".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Real,
            },
            Operators::Div => Operator {
                id: 4,
                name: "div".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Real,
            },
            Operators::And => Operator {
                id: 5,
                name: "and".to_string(),
                arity: 2,
                input_type: NodeType::Bool,
                output_type: NodeType::Bool,
            },
            Operators::Or => Operator {
                id: 6,
                name: "or".to_string(),
                arity: 2,
                input_type: NodeType::Bool,
                output_type: NodeType::Bool,
            },
            Operators::Not => Operator {
                id: 7,
                name: "not".to_string(),
                arity: 1,
                input_type: NodeType::Bool,
                output_type: NodeType::Bool,
            },
            Operators::Gt => Operator {
                id: 8,
                name: "gt".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Bool,
            },
            Operators::Ge => Operator {
                id: 9,
                name: "ge".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Bool,
            },
            Operators::Lt => Operator {
                id: 10,
                name: "lt".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Bool,
            },
            Operators::Le => Operator {
                id: 11,
                name: "le".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Bool,
            },
            Operators::Eq => Operator {
                id: 12,
                name: "eq".to_string(),
                arity: 2,
                input_type: NodeType::Any,
                output_type: NodeType::Bool,
            },
        }
    }
}

#[derive(Clone)]
pub enum ParsingNode {
    Feature(String),
    Operator(Operators),
    Constant(f32),
}

impl From<ParsingNode> for Node {
    fn from(parsing_node: ParsingNode) -> Self {
        match parsing_node {
            ParsingNode::Feature(field_code) => Node::Feature(Feature {
                id: 0,
                code: field_code,
                name: "".to_string(),
                node_type: NodeType::Any,
                cohort_id: 0,
            }),
            ParsingNode::Operator(operator) => Node::Operator(operator),
            ParsingNode::Constant(value) => Node::Constant(Constant {
                value,
                node_type: NodeType::Real,
            }),
        }
    }
}
