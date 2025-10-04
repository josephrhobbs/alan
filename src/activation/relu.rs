//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! ReLU activation function.

use crate::{
    activation::Activation,
    Numeric,
    tensor::Batch,
};

pub struct ReLU<const N: usize>;

impl<const B: usize, T: Numeric, const N: usize> Activation<B, T, N> for ReLU<N> {
    fn new() -> Self {
        Self
    }

    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, N> {
        let mut result = Batch::<B, T, N>::zero();

        for b in 0..B {
            for i in 0..N {
                result[b][i] = if batch[b][i] >= T::zero() {
                    batch[b][i]
                } else {
                    T::zero()
                };
            }
        }

        result
    }

    fn backward(&self, batch: &Batch<B, T, N>) -> Batch<B, T, N> {
        let mut result = Batch::<B, T, N>::zero();

        for b in 0..B {
            for i in 0..N {
                result[b][i] = if batch[b][i] != T::zero() {
                    T::one()
                } else {
                    T::zero()
                };
            }
        }

        result
    }
}
