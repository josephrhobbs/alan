//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Softmax activation function.

use crate::{
    activation::Activation,
    Numeric,
    tensor::Batch,
};

pub struct Softmax<const B: usize, T: Numeric, const N: usize> {
    output: Batch<B, T, N>,
}

impl<const B: usize, T: Numeric, const N: usize> Activation<B, T, N> for Softmax<B, T, N> {
    fn new() -> Self {
        Self {
            output: Batch::zero(),
        }
    }

    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, N> {
        let mut result = Batch::<B, T, N>::zero();
        let mut denom = [T::zero(); B];
        let mut maxval = [T::zero(); B];

        // TODO what about integers?

        // Store max value in each batch
        for b in 0..B {
            for i in 0..N {
                if batch[b][i] > maxval[b] {
                    maxval[b] = batch[b][i];
                }
            }
        }

        // Compute denominator
        for b in 0..B {
            for i in 0..N {
                denom[b] = denom[b] + T::exp(batch[b][i] - maxval[b]);
                
            }
        }

        for b in 0..B {
            for i in 0..N {
                result[b][i] = T::exp(batch[b][i] - maxval[b]) / denom[b];
            }
        }

        self.output = result;

        result
    }

    fn backward(&self, batch: &Batch<B, T, N>) -> Batch<B, T, N> {
        let mut result = Batch::<B, T, N>::zero();

        for b in 0..B {
            for i in 0..N {
                for j in 0..N {
                    let delta = if i == j {
                        T::one()
                    } else {
                        T::zero()
                    };
                    result[b][j] = result[b][j] + batch[b][i] * self.output[b][i] * (delta - self.output[b][j]);
                }
            }
        }

        dbg!(batch, result, self.output);

        result
    }
}
