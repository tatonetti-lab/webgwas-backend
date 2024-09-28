use anyhow::{bail, Result};
use itertools::izip;
use polars::prelude::*;
use std::{path::PathBuf, sync::Arc};

use crate::models::{ChromosomePosition, Pvalue, PvaluesResult};

fn read_pvalue_df(path: PathBuf) -> Result<DataFrame> {
    let schema_override = Schema::from_iter(vec![("neg_log_p_value".into(), DataType::Float32)]);
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
        _ => chrom.parse::<i32>().expect("Failed to parse chromosome"),
    }
}

fn index_to_chrom(index: i32) -> String {
    match index {
        22 => "X".to_string(),
        23 => "Y".to_string(),
        _ => format!("{}", index),
    }
}

/// Create a column that indexes the chromosomes in order (1, 2, ..., 22, X, Y)
fn build_chromosome_index(df: &DataFrame) -> Result<Column> {
    let mut result: Column = df
        .column("chromosome")?
        .str()?
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

#[derive(Copy, Clone)]
struct ChromosomeBounds {
    pub min: i32,
    pub max: i32,
}

fn extract_chromosome_positions(pvalues: &[Pvalue]) -> Result<Vec<ChromosomePosition>> {
    let mut chromosomes = [ChromosomeBounds {
        min: 0,
        max: i32::MAX,
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
    for x in chromosomes.iter() {
        if x.min == 0 || x.max == i32::MAX {
            bail!("Failed to extract chromosome positions");
        }
    }
    let result = chromosomes
        .iter()
        .map(|x| ChromosomePosition {
            chromosome: index_to_chrom(x.min),
            midpoint: (x.min + x.max) / 2,
        })
        .collect();
    Ok(result)
}

/// Load p-values from a result file
pub fn load_pvalues(path: PathBuf) -> Result<PvaluesResult> {
    let mut df = read_pvalue_df(path)?;
    let index_col = build_chromosome_index(&df)?;
    df.with_column(index_col)?.sort(
        ["chromosome_index".to_string(), "position".to_string()],
        Default::default(),
    )?;
    let pvalues = extract_pvalue_vec(&df)?;
    let chromosome_positions = extract_chromosome_positions(&pvalues)?;
    Ok(PvaluesResult {
        pvalues,
        chromosome_positions,
    })
}
