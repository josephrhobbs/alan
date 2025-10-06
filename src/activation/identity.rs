//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Identity activation function.

use crate::{
    activation::Activation,
    Numeric,
    tensor::Batch,
};

pub struct Identity<const N: usize>;

impl<const B: usize, T: Numeric, const N: usize> Activation<B, T, N> for Identity<N> {
    fn new() -> Self {
        Self
    }

    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, N> {
        *batch
    }

    fn backward(&self, batch: &Batch<B, T, N>) -> Batch<B, T, N> {
        *batch
    }
}
