use anyhow::{Context, Result};
use itertools::izip;
use polars::prelude::*;
use std::{path::PathBuf, sync::Arc};

use crate::models::{ChromosomePosition, Pvalue, PvaluesResult};

fn read_pvalue_df(path: PathBuf) -> Result<DataFrame> {
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
fn build_chromosome_index(df: &DataFrame) -> Result<Column> {
    let mut result: Column = df
        .column("chromosome")?
        .str()
        .context("Chromosome column is not str")?
        .clone()
        .iter()
        .map(|chr_opt| chr_opt.map(chrom_to_index))
        .collect::<Int32Chunked>()
        .into_column();
    result.rename("chromosome_index".into());
    Ok(result)
}

fn extract_pvalue_vec(df: &DataFrame) -> Result<Vec<Pvalue>> {
    let pvalues: Vec<Pvalue> = izip!(
        df.column("neg_log_p_value")?.f32()?.into_iter(),
        df.column("chromosome")?.str()?.into_iter(),
        df.column("rsid")?.str()?.into_iter(),
    )
    .enumerate()
    .map(|(i, (pvalue, chromosome, rsid))| Pvalue {
        index: i as i32,
        pvalue: pvalue.expect("Failed to get pvalue"),
        chromosome: chromosome.expect("Failed to get chromosome").to_string(),
        label: rsid.expect("Failed to get rsid").to_string(),
    })
    .collect();
    Ok(pvalues)
}

#[derive(Copy, Clone, Debug)]
struct ChromosomeBounds {
    pub min: i32,
    pub max: i32,
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

/// Load p-values from a result file
pub fn load_pvalues(path: PathBuf, min_neg_log_p: Option<f32>) -> Result<PvaluesResult> {
    let mut df = read_pvalue_df(path).context("Failed to read p-value df")?;
    if let Some(min_neg_log_p) = min_neg_log_p {
        df = df
            .lazy()
            .filter(col("neg_log_p_value").gt_eq(min_neg_log_p))
            .collect()?;
    }
    let index_col = build_chromosome_index(&df).context("Failed to build chromosome index")?;
    let df = df.with_column(index_col)?.sort(
        ["chromosome_index".to_string(), "position".to_string()],
        SortMultipleOptions::default(),
    )?;
    let pvalues = extract_pvalue_vec(&df).context("Failed to extract p-values")?;
    let chromosome_positions = extract_chromosome_positions(&pvalues);
    Ok(PvaluesResult {
        pvalues,
        chromosome_positions,
    })
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
                label: "rs1".to_string(),
            },
            Pvalue {
                index: 1,
                pvalue: 0.2,
                chromosome: "1".to_string(),
                label: "rs2".to_string(),
            },
            Pvalue {
                index: 2,
                pvalue: 0.3,
                chromosome: "Y".to_string(),
                label: "rs3".to_string(),
            },
            Pvalue {
                index: 3,
                pvalue: 0.4,
                chromosome: "Y".to_string(),
                label: "rs4".to_string(),
            },
        ];
        let result = extract_chromosome_positions(&pvalues);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chromosome, "1");
        assert_eq!(result[1].chromosome, "Y");
    }
}
