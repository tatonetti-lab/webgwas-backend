use anyhow::{bail, Result};
use faer::{Col, Mat};

pub fn add_intercept(x: &mut Mat<f32>) {
    x.resize_with(x.nrows(), x.ncols() + 1, |_, _| 1.0);
}

pub fn count_nans(x: &Mat<f32>) -> usize {
    let mut n_nans = 0;
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            if x[(i, j)].is_nan() {
                n_nans += 1;
            }
        }
    }
    n_nans
}

pub fn compute_left_inverse(x: &Mat<f32>) -> Result<Mat<f32>> {
    let result = x.thin_svd().pseudoinverse();
    if count_nans(&result) > 0 {
        bail!("Failed to compute left inverse");
    }
    Ok(result)
}

pub fn regress_left_inverse_vec(endog: &Col<f32>, exog_left_inverse: &Mat<f32>) -> Col<f32> {
    exog_left_inverse * endog
}

pub fn regress_left_inverse_mat(endog: &Mat<f32>, exog_left_inverse: &Mat<f32>) -> Mat<f32> {
    exog_left_inverse * endog
}

pub fn regress_vec(endog: &Col<f32>, exog: &Mat<f32>) -> Result<Col<f32>> {
    let left_inverse = compute_left_inverse(exog)?;
    let result = regress_left_inverse_vec(endog, &left_inverse);
    Ok(result)
}

pub fn regress_mat(endog: &Mat<f32>, exog: &Mat<f32>) -> Result<Mat<f32>> {
    let left_inverse = compute_left_inverse(exog)?;
    let result = regress_left_inverse_mat(endog, &left_inverse);
    Ok(result)
}

pub fn compute_covariance(x: &Mat<f32>, ddof: usize) -> Mat<f32> {
    // Normalize each column to mean zero
    let mut x_norm = x.clone();
    for i in 0..x.ncols() {
        let col_mean = x.col(i).sum() / (x.nrows() as f32);
        x_norm.col_mut(i).iter_mut().for_each(|x| *x -= col_mean);
    }
    x_norm.clone().transpose() * x_norm / (x.nrows() - ddof) as f32
}

pub fn residualize_covariates(x: &Mat<f32>, y: &Mat<f32>) -> Result<Mat<f32>> {
    Ok(y - x * &regress_mat(y, x)?)
}

pub fn transpose_vec_vec<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{col, mat};

    #[test]
    fn test_regress() {
        let x = mat![
            [1.0, 2.0, -1.0, 2.5],
            [1.5, 3.3, -0.5, 2.0],
            [3.1, 0.7, 2.2, 0.0],
            [0.0, 0.3, -2.0, 5.3],
            [2.1, 1.0, 4.3, 2.2],
            [0.0, 5.5, 3.8, 0.2]
        ];
        let y = col![0.0, 1.0, 5.3, -2.0, 6.3, 3.8];
        let result = regress_vec(&y, &x).unwrap();
        let expected: Col<f32> = col![1.0009506, 0.00881193, 0.98417587, -0.01009547];
        assert!((result - expected).squared_norm_l2() < 1e-6);
    }

    #[test]
    fn test_covariance() {
        let x = mat![
            [1.0, 2.0, -1.0, 2.5],
            [1.5, 3.3, -0.5, 2.0],
            [3.1, 0.7, 2.2, 0.0],
            [0.0, 0.3, -2.0, 5.3],
            [2.1, 1.0, 4.3, 2.2],
            [0.0, 5.5, 3.8, 0.2]
        ];
        let result = compute_covariance(&x, 1);
        let expected: Mat<f32> = mat![
            [1.4776666, -1.0413333, 1.0746666, -1.1073334],
            [-1.0413333, 3.8826668, 1.5966663, -1.9073334],
            [1.0746666, 1.5966663, 7.0626664, -3.5413334],
            [-1.1073334, -1.9073334, -3.5413334, 3.6826668]
        ];
        assert!((result - expected).squared_norm_l2() < 1e-6);
    }

    #[test]
    fn test_covariance_ddof_0() {
        let x = mat![
            [1.0, 2.0, -1.0, 2.5],
            [1.5, 3.3, -0.5, 2.0],
            [3.1, 0.7, 2.2, 0.0],
            [0.0, 0.3, -2.0, 5.3],
            [2.1, 1.0, 4.3, 2.2],
            [0.0, 5.5, 3.8, 0.2]
        ];
        let result = compute_covariance(&x, 0);
        let expected: Mat<f32> = mat![
            [1.2313889, -0.86777776, 0.89555556, -0.9227778],
            [-0.86777776, 3.2355556, 1.3305556, -1.5894444],
            [0.89555556, 1.3305556, 5.8855557, -2.951111],
            [-0.9227778, -1.5894444, -2.951111, 3.068889]
        ];
        assert!((result - expected).squared_norm_l2() < 1e-6);
    }

    #[test]
    fn test_covariance_ddof_3() {
        let x = mat![
            [1.0, 2.0, -1.0, 2.5],
            [1.5, 3.3, -0.5, 2.0],
            [3.1, 0.7, 2.2, 0.0],
            [0.0, 0.3, -2.0, 5.3],
            [2.1, 1.0, 4.3, 2.2],
            [0.0, 5.5, 3.8, 0.2]
        ];
        let result = compute_covariance(&x, 3);
        let expected: Mat<f32> = mat![
            [2.4627779, -1.7355555, 1.7911111, -1.8455555],
            [-1.7355555, 6.4711113, 2.661111, -3.1788888],
            [1.7911111, 2.661111, 11.7711115, -5.902222],
            [-1.8455555, -3.1788888, -5.902222, 6.137778]
        ];
        assert!((result - expected).squared_norm_l2() < 1e-6);
    }

    #[test]
    fn test_resize() {
        let mut x: Mat<f32> = mat![[1.0, 2.0], [1.5, 3.3],];
        add_intercept(&mut x);
        let expected: Mat<f32> = mat![[1.0, 2.0, 1.0], [1.5, 3.3, 1.0]];
        assert!((x - expected).squared_norm_l2() < 1e-6);
    }
}
