use anyhow::Result;
use faer::Col;
use polars::series::Series;

/// Get everything up to and including the item
pub fn slice_before<T: PartialEq + Clone>(vec: &[T], item: &T) -> Vec<T> {
    if let Some(index) = vec.iter().position(|x| x == item) {
        vec[0..index].to_vec()
    } else {
        Vec::new()
    }
}

/// Get everything after and excluding the item
pub fn slice_after_excl<T: PartialEq + Clone>(vec: &[T], item: &T) -> Vec<T> {
    if let Some(index) = vec.iter().position(|x| x == item) {
        vec[index + 1..].to_vec()
    } else {
        vec.to_vec()
    }
}

/// Get everything up to but excluding the item
pub fn slice_before_excl<T: PartialEq + Clone>(vec: &[T], item: &T) -> Vec<T> {
    if let Some(index) = vec.iter().position(|x| x == item) {
        vec[0..index].to_vec()
    } else {
        vec.to_vec()
    }
}

/// Convert a polars Series to a faer Col
pub fn series_to_col_vector(series: Series) -> Result<Col<f32>> {
    let mut result = Col::zeros(series.len());
    series.f32()?.iter().enumerate().for_each(|(i, x)| {
        result[i] = x.expect("Failed to get value");
    });
    Ok(result)
}
