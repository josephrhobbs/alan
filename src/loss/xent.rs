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
/// 
/// NOTE this layer accepts _raw logits_ instead of
/// class probabilities as computed with softmax.
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
        self.labels = *labels;

        // Initialize result
        let mut loss = T::zero();
        let mut b_as_t = T::zero();

        // Store max value in each batch
        let mut maxval = [T::neginf(); B];
        for b in 0..B {
            for i in 0..N {
                if prediction[b][i] > maxval[b] {
                    maxval[b] = prediction[b][i];
                }
            }
        }

        // Compute denominator
        let mut denom = [T::zero(); B];
        for b in 0..B {
            for i in 0..N {
                denom[b] = denom[b] + T::exp(prediction[b][i] - maxval[b]);
            }
        }

        for b in 0..B {
            for i in 0..N {
                self.prediction[b][i] = T::exp(prediction[b][i] - maxval[b]) / denom[b];
                loss = loss - labels[b][i] * (prediction[b][i] - maxval[b] - T::log(denom[b]));
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
                backward[b][i] = self.prediction[b][i] - self.labels[b][i];
            }
        }

        backward
    }
}
