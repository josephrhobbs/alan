//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Linear regressor.

use crate::{
    architecture::Architecture,
    network::{
        activation::Identity,
        layer::Linear,
        Layer,
    },
    Numeric,
    optim::loss::MSELoss,
    tensors::Batch,
};

/// Single variable linear regression model.
pub struct LinearRegressor<const B: usize, T: Numeric> {
    linear_layer: Linear<B, T, 1, 1>,
}

/// Training and inference implementation for this regressor.
impl<const B: usize, T: Numeric> Architecture<B, T, 1, 1> for LinearRegressor<B, T> {
    /// Mean squared error loss.
    type LossFunction = MSELoss<B, T, 1>;

    /// Identity activation
    type Activation = Identity<1>;

    /// Construct this regressor.
    fn new() -> Self {
        Self {
            linear_layer: Linear::new(),
        }
    }

    /// Compute the forward pass of this regressor.
    fn forward(&mut self, batch: &Batch<B, T, 1>) -> Batch<B, T, 1> {
        self.linear_layer.forward(batch)
    }

    /// Compute the backward pass of this regressor.
    fn backward(&mut self, gradients: &Batch<B, T, 1>, lr: T) {
        self.linear_layer.backward(gradients, lr);
    }
}
