//! ALAN
//! Autodifferentiation Library with Applications to Neural networks
//! Copyright (c) 2025 J. Hobbs
//!
//! Main library.

mod layer;
mod numeric;
mod tensors;

pub use crate::numeric::Numeric;

pub mod network {
    pub use crate::layer::Layer;
    pub use crate::layer::Linear;
    pub use crate::layer::ReLU;
}

pub mod tensor {
    pub use crate::tensors::Tensor;
    pub use crate::tensors::Batch;
}