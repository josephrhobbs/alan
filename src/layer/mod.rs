//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Network layer abstraction.

mod convolutional;
mod linear;

use crate::{
    Numeric,
    tensor::Batch,
};

pub use linear::Linear;

/// Network layer abstraction.
pub trait Layer<const B: usize, T: Numeric, const N: usize, const M: usize> {
    /// Construct a new network layer.
    fn new() -> Self;

    /// Complete a forward pass through this layer.
    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, M>;

    /// Complete a backward pass through this layer.
    fn backward(&mut self, batch: &Batch<B, T, M>, lr: T) -> Batch<B, T, N>;
}
