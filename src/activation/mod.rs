//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Network activation function abstraction.

mod relu;
mod softmax;

use crate::{
    Numeric,
    tensor::Batch,
};

pub use relu::ReLU;
pub use softmax::Softmax;

/// Network activation function abstraction.
pub trait Activation<const B: usize, T: Numeric, const N: usize> {
    /// Construct a new network activation function.
    fn new() -> Self;

    /// Complete a forward pass through this activation function.
    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, N>;

    /// Complete a backward pass through this activation function.
    fn backward(&self, batch: &Batch<B, T, N>) -> Batch<B, T, N>;
}
