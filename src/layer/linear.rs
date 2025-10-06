//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Linear network layer.

use crate::{
    layer::Layer,
    Numeric,
    tensor::Batch,
};

/// Linear network layer.
/// 
/// A linear layer `Linear<B, T, N, M>` maps `Tensor`s of size `N` and type `T`
/// to `Tensor`s of size `M` and identical type.  `Tensor`s are evaluated in
/// `Batch`es to improve performance.
pub struct Linear<const B: usize, T: Numeric, const N: usize, const M: usize> {
    /// Last network layer input.
    input: Batch<B, T, N>,
    
    /// Layer weights.
    weights: [[T; N]; M],

    /// Layer bias.
    bias: [T; M],
}

impl<const B: usize, T: Numeric, const N: usize, const M: usize> Layer<B, T, N, M> for Linear<B, T, N, M> {
    /// Construct a new layer, setting all parameters randomly.
    fn new() -> Self {
        // Initialize parameters randomly
        let mut weights = [[T::zero(); N]; M];
        let mut bias    = [T::zero(); M];
        for i in 0..M {
            for j in 0..N {
                weights[i][j] = T::random();
            }
            bias[i] = T::random();
        }

        Self {
            input: Batch::<B, T, N>::zero(),
            weights,
            bias,
        }
    }
    
    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, M> {
        self.input = *batch;

        let mut result = Batch::<B, T, M>::zero();

        for b in 0..B {
            let input = batch[b];

            for i in 0..M {
                for j in 0..N {
                    result[b][i] = result[b][i] + self.weights[i][j] * input[j];
                }
                result[b][i] = result[b][i] + self.bias[i];
            }
        }

        result
    }

    fn backward(&mut self, batch: &Batch<B, T, M>, lr: T) -> Batch<B, T, N> {
        // Backpropagate gradients
        let mut backward = Batch::<B, T, N>::zero();
        let mut b_as_t = T::zero();
        for b in 0..B {
            for i in 0..M {
                for j in 0..N {
                    backward[b][j] = backward[b][j] + batch[b][i] * self.weights[i][j];
                }
            }
            b_as_t = b_as_t + T::one();
        }

        // Update network parameters using batch gradient descent
        // TODO generalize to different types of GD (e.g. SGD, Adam)
        for b in 0..B {
            for i in 0..M {
                for j in 0..N {
                    self.weights[i][j] = self.weights[i][j] - batch[b][i] * self.input[b][j] * lr * T::one() / b_as_t;
                }
                self.bias[i] = self.bias[i] - batch[b][i] * lr * T::one() / b_as_t;
            }
        }

        backward
    }
}
