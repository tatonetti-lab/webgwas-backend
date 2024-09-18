use faer::{Col, Mat};

pub fn regress_left_inverse(endog: &Col<f32>, exog_left_inverse: &Mat<f32>) -> Col<f32> {
    exog_left_inverse * endog
}

pub fn regress_vec(endog: &Col<f32>, exog: &Mat<f32>) -> Col<f32> {
    let svd = exog.thin_svd();
    let mut s = svd.s_diagonal().to_owned();
    s.iter_mut().for_each(|x| *x = x.recip());
    let s_inv_mat = s.column_vector_as_diagonal();
    let result = svd.v() * s_inv_mat * svd.u().transpose() * endog;
    result
}

pub fn regress_mat(endog: &Mat<f32>, exog: &Mat<f32>) -> Mat<f32> {
    let svd = exog.thin_svd();
    let mut s = svd.s_diagonal().to_owned();
    s.iter_mut().for_each(|x| *x = x.recip());
    let s_inv_mat = s.column_vector_as_diagonal();
    svd.v() * s_inv_mat * svd.u().transpose() * endog
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
        let result = regress_vec(&y, &x);
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
}
