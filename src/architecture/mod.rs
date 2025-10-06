//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Network architecture abstraction.

pub mod classifiers;
pub mod regressors;

use crate::{
    network::Activation,
    Numeric,
    optim::{
        Hyperparameters,
        Loss,
    },
    tensor::{
        Batch,
        Dataset,
    },
};

/// Network architecture abstraction.
/// 
/// NOTE types that are `Architecture` have distinct `Architecture::forward` and
/// `Architecture::eval` methods.  The `forward` method computes a "raw" network
/// output over which to optimize.  The `eval` method computes a "processed" network
/// output for later use.  The `eval` method calls the `Activation` defined in the trait.
/// To keep these the same, use the `alan::network::activation::Identity` activation.
pub trait Architecture<const B: usize, T: Numeric, const N: usize, const M: usize> {
    /// Loss function for this architecture.
    type LossFunction: Loss<B, T, M>;

    /// Activation function for the output of this model.
    type Activation: Activation<B, T, M>;

    /// Initialize this architecture.
    fn new() -> Self;

    /// Compute the forward pass of this model, returning an inference result.
    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, M>;

    /// Evaluate this model, applying the given activation.
    fn eval(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, M> {
        let mut activation = Self::Activation::new();
        activation.forward(&self.forward(batch))
    }

    /// Compute the backward pass of this model, updating all parameters.
    fn backward(&mut self, gradients: &Batch<B, T, M>, lr: T);

    /// Train this network on a given training set using the provided
    /// hyperparameters, and return the loss from each epoch.
    fn train(&mut self, dataset: &mut Dataset<B, T, N, M>, hyperparameters: Hyperparameters<T>) -> Vec<T> {
        // Instantiate loss function
        let mut loss_function = Self::LossFunction::new();

        // Store losses
        let mut losses = Vec::new();

        for _ in 0..hyperparameters.epochs {
            // Total loss for this epoch
            let mut total_loss = T::zero();

            while let Some ((data, labels)) = dataset.next() {
                // Compute forward pass
                let result = self.forward(&data); 

                // Compute loss
                let loss = loss_function.forward(&result, &labels);
                total_loss = total_loss + loss;
                let gradients = loss_function.backward();

                // Compute backward pass
                self.backward(&gradients, hyperparameters.lr);
            }

            // Store loss
            losses.push(total_loss);

            // Refresh dataset
            dataset.refresh();
        }

        losses
    }

    /// Compute the loss over a test dataset.
    fn test(&mut self, dataset: &mut Dataset<B, T, N, M>) -> T {
        // Count number of batches
        let mut batches_as_t = T::zero();

        // Total loss
        let mut total_loss = T::zero();

        // Instantiate loss function
        let mut loss_function = Self::LossFunction::new();

        while let Some ((data, labels)) = dataset.next() {
            // Compute loss for this batch
            let prediction = self.forward(&data);
            total_loss = total_loss + loss_function.forward(&prediction, &labels);

            // Increment batch count
            batches_as_t = batches_as_t + T::one();
        }

        // Refresh dataset
        dataset.refresh();

        total_loss / batches_as_t
    }
}
