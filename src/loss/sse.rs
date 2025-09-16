//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Sum of squared error loss function.

use crate::{
    loss::Loss,
    Numeric,
    tensor::Batch,
};

/// Sum of squared error loss layer.
pub struct SSELoss<const B: usize, T: Numeric, const N: usize> {
    prediction: Batch<B, T, N>,
    labels: Batch<B, T, N>,
}

impl<const B: usize, T: Numeric, const N: usize> Loss<B, T, N> for SSELoss<B, T, N> {
    fn new() -> Self {
        Self {
            prediction: Batch::zero(),
            labels: Batch::zero(),
        }
    }

    fn forward(&mut self, prediction: Batch<B, T, N>, labels: Batch<B, T, N>) -> T {
        self.prediction = prediction;
        self.labels = labels;

        // Initialize result
        let mut loss = T::zero();

        for b in 0..B {
            for i in 0..N {
                loss = loss + (prediction[b][i] - labels[b][i]) * (prediction[b][i] - labels[b][i]);
            }
        }

        loss
    }

    fn backward(&self) -> Batch<B, T, N> {
        // Backpropagate gradients
        let mut backward = Batch::<B, T, N>::zero();
        let two = T::one() + T::one();
        for b in 0..B {
            for i in 0..N {
                backward[b][i] = two * (self.prediction[b][i] - self.labels[b][i]);
            }
        }

        backward
    }
}
