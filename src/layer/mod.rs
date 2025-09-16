//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Network layer abstraction.

mod linear;
mod relu;

use crate::{
    Numeric,
    tensor::Batch,
};

pub use linear::Linear;
pub use relu::ReLU;

pub trait Layer<const B: usize, T: Numeric, const N: usize, const M: usize> {
    fn forward(&mut self, batch: Batch<B, T, N>) -> Batch<B, T, M>;

    fn backward(&mut self, batch: Batch<B, T, M>, lr: T) -> Batch<B, T, N>;
}

