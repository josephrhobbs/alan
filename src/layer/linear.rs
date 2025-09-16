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
    
    /// Layer parameters.
    parameters: [[T; N]; M],
}

impl<const B: usize, T: Numeric, const N: usize, const M: usize> Layer<B, T, N, M> for Linear<B, T, N, M> {
    /// Construct a new layer, setting all parameters to zero.
    fn new() -> Self {
        Self {
            input: Batch::<B, T, N>::zero(),
            parameters: [[T::one(); N]; M],
        }
    }
    
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
        // Backpropagate gradients
        let mut backward = Batch::<B, T, N>::zero();
        for b in 0..B {
            for i in 0..M {
                for j in 0..N {
                    backward[b][j] = backward[b][j] + batch[b][i] * self.parameters[i][j];
                }
            }
        }

        // Update network parameters using batch gradient descent
        // TODO generalize to different types of GD (e.g. SGD, Adam)
        for b in 0..B {
            for i in 0..M {
                for j in 0..N {
                    self.parameters[i][j] = self.parameters[i][j] - batch[b][i] * self.input[b][j] * lr
                }
            }
        }

        backward
    }
}
