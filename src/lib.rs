//! ALAN
//! Autodifferentiation Library with Applications to Neural networks
//! Copyright (c) 2025 J. Hobbs
//!
//! Main library.

mod activation;
mod architecture;
mod dataset;
mod hyperparameters;
mod layer;
mod loss;
mod numeric;
mod tensors;

pub use crate::numeric::Numeric;

pub use crate::numeric::x16;

pub mod models {
    pub use crate::architecture::Architecture;

    // Predefined classifiers
    pub use crate::architecture::classifiers;

    // Predefined regressors
    pub use crate::architecture::regressors;
}

pub mod network {
    pub use crate::activation::Activation;

    pub mod activation {
        pub use crate::activation::Identity;
        pub use crate::activation::ReLU;
        pub use crate::activation::Softmax;
    }

    pub use crate::layer::Layer;

    pub mod layer {
        pub use crate::layer::AvgPool;
        pub use crate::layer::Convolution;
        pub use crate::layer::Linear;
    }
}

pub mod optim {
    pub use crate::hyperparameters::Hyperparameters;

    pub use crate::loss::Loss;

    pub mod loss {
        pub use crate::loss::CrossEntropyLoss;
        pub use crate::loss::MSELoss;
    }
}

pub mod tensor {
    pub use crate::dataset::Dataset;
    pub use crate::tensors::Tensor;
    pub use crate::tensors::Batch;
}
