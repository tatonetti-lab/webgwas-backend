use ndarray::{Array1, Array2};

pub fn regress(endog: &Array1<f32>, exog_left_inverse: &Array2<f32>) -> Array1<f32> {
    exog_left_inverse.dot(endog)
}
