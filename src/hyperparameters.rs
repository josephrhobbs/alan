//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Hyperparameters abstraction.

use crate::Numeric;

#[derive(Clone, Copy, Debug)]
/// Hyperparameters for model training.
///
/// As of this writing, available hyperparameters are:
/// - `epochs: usize`: training epochs
/// - `lr: T`: learning rate
pub struct Hyperparameters<T: Numeric> {
    pub epochs: usize,
    pub lr: T,
}
