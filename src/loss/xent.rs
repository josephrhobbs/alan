//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Cross-entropy loss function.

use crate::{
    loss::Loss,
    Numeric,
    tensor::Batch,
};

/// Cross-entropy loss layer.
pub struct CrossEntropyLoss<const B: usize, T: Numeric, const N: usize> {
    prediction: Batch<B, T, N>,
    labels: Batch<B, T, N>,
}

impl<const B: usize, T: Numeric, const N: usize> Loss<B, T, N> for CrossEntropyLoss<B, T, N> {
    fn new() -> Self {
        Self {
            prediction: Batch::zero(),
            labels: Batch::zero(),
        }
    }

    fn forward(&mut self, prediction: &Batch<B, T, N>, labels: &Batch<B, T, N>) -> T {
        self.prediction = *prediction;
        self.labels = *labels;

        // Initialize result
        let mut loss = T::zero();
        let mut b_as_t = T::zero();

        for b in 0..B {
            for i in 0..N {
                loss = loss - labels[b][i] * T::log(prediction[b][i]);
            }
            b_as_t = b_as_t + T::one();
        }

        loss / b_as_t
    }

    fn backward(&self) -> Batch<B, T, N> {
        // Backpropagate gradients
        let mut backward = Batch::<B, T, N>::zero();
        for b in 0..B {
            for i in 0..N {
                backward[b][i] = - self.labels[b][i] / self.prediction[b][i];
            }
        }

        backward
    }
}
