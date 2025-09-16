//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! ReLU activation layer.

use crate::{
    layer::Layer,
    Numeric,
    tensor::Batch,
};

pub struct ReLU<const N: usize>;

impl<const B: usize, T: Numeric, const N: usize> Layer<B, T, N, N> for ReLU<N> {
    fn new() -> Self {
        Self
    }

    fn forward(&mut self, batch: Batch<B, T, N>) -> Batch<B, T, N> {
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

    fn backward(&mut self, batch: Batch<B, T, N>, _lr: T) -> Batch<B, T, N> {
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
