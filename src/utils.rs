/// Get everything after and including the item
pub fn slice_after<T: PartialEq + Clone>(vec: &[T], item: &T) -> Vec<T> {
    if let Some(index) = vec.iter().position(|x| x == item) {
        vec[index..].to_vec()
    } else {
        Vec::new()
    }
}

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
