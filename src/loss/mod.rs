//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Network loss function abstraction.

mod mse;
mod xent;

use crate::{
    Numeric,
    tensor::Batch,
};

pub use mse::MSELoss;
pub use xent::CrossEntropyLoss;

/// Network loss function abstraction.
pub trait Loss<const B: usize, T: Numeric, const N: usize> {
    /// Construct a new loss function.
    fn new() -> Self;

    /// Complete a forward pass through this layer.
    fn forward(&mut self, prediction: Batch<B, T, N>, labels: Batch<B, T, N>) -> T;

    /// Complete a backward pass through this layer.
    fn backward(&self) -> Batch<B, T, N>;
}