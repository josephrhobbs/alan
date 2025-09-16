//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Linear network layer.

use crate::{
    layer::Layer,
    Numeric,
    tensor::Batch,
};

pub struct Linear<const B: usize, T: Numeric, const N: usize, const M: usize> {
    pub input: Batch<B, T, N>,
    pub parameters: [[T; N]; M],
}

impl<const B: usize, T: Numeric, const N: usize, const M: usize> Layer<B, T, N, M> for Linear<B, T, N, M> {
    fn forward(&mut self, batch: Batch<B, T, N>) -> Batch<B, T, M> {
        self.input = batch;

        let mut result = Batch::<B, T, M>::zero();

        for b in 0..B {
            let input = batch[b];

            for i in 0..M {
                for j in 0..N {
                    result[b][i] = result[b][i] + self.parameters[i][j] * input[j];
                }
            }
        }

        result
    }

    fn backward(&mut self, batch: Batch<B, T, M>, lr: T) -> Batch<B, T, N> {
        for b in 0..B {
            for i in 0..M {
                for j in 0..N {
                    self.parameters[i][j] = self.parameters[i][j] - batch[b][i] * self.input[b][j] * lr
                }
            }
        }

        let mut backward = Batch::<B, T, N>::zero();

        // TODO backpropagate!

        backward
    }
}
