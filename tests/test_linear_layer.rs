//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Linear regression testbench.

use alan::{
    models::{
        Architecture,
        regressors::LinearRegressor,
    },
    optim::Hyperparameters,
    tensor::{
        Dataset,
        Tensor,
    },
};

#[test]
fn test_linear_regressor() {
    let data = vec![
        Tensor::<f64, 1> ([0.0]),
        Tensor::<f64, 1> ([1.0]),
        Tensor::<f64, 1> ([2.0]),
        Tensor::<f64, 1> ([3.0]),
    ];

    let labels = vec![
        Tensor::<f64, 1> ([0.0]),
        Tensor::<f64, 1> ([2.0]),
        Tensor::<f64, 1> ([4.0]),
        Tensor::<f64, 1> ([6.0]),
    ];

    // Initialize dataset
    let dataset = Dataset::<4, f64, 1, 1>::new(data, labels);
    assert!(dataset.is_some());
    let mut dataset = dataset.unwrap();

    // Initialize model
    let mut model = LinearRegressor::<4, f64>::new();

    // Set up hyperparameters
    let h = Hyperparameters {epochs: 25, lr: 0.05};

    // Train model
    model.train(&mut dataset, h);

    // New data!
    let data = vec![
        Tensor::<f64, 1> ([4.0]),
        Tensor::<f64, 1> ([5.0]),
        Tensor::<f64, 1> ([6.0]),
        Tensor::<f64, 1> ([7.0]),
    ];
    let labels = vec![
        Tensor::<f64, 1> ([8.0]),
        Tensor::<f64, 1> ([10.0]),
        Tensor::<f64, 1> ([12.0]),
        Tensor::<f64, 1> ([14.0]),
    ];
    let dataset = Dataset::<4, f64, 1, 1>::new(data, labels);
    assert!(dataset.is_some());
    let mut dataset = dataset.unwrap();

    // Check loss on new data
    let loss = model.test(&mut dataset);
    assert!(loss < 1e-3);
}
