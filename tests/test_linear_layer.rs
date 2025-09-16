//! ALAN Testbench

use alan::{
    tensor::{
        Batch,
        Tensor,
    },
    network::{
        Layer,
        Activation,
        layer::Linear,
        activation::Softmax,
    },
    optim::{
        Loss,
        loss::MSELoss,
        loss::CrossEntropyLoss,
    },
};

#[test]
fn linear_regressor() {
    let batch = Batch::<4, f64, 2> ([
        Tensor::<f64, 2> ([0.0, 1.0]),
        Tensor::<f64, 2> ([1.0, 1.0]),
        Tensor::<f64, 2> ([2.0, 1.0]),
        Tensor::<f64, 2> ([3.0, 1.0]),
    ]);

    let labels = Batch::<4, f64, 2> ([
        Tensor::<f64, 2> ([1.0, 1.0]),
        Tensor::<f64, 2> ([1.5, 1.0]),
        Tensor::<f64, 2> ([2.0, 1.0]),
        Tensor::<f64, 2> ([2.5, 1.0]),
    ]);

    // Initialize linear layer
    let mut linear_layer = Linear::<4, f64, 2, 2>::new();

    // Initialize loss function
    let mut loss = MSELoss::<4, f64, 2>::new();

    for _ in 0..25 {
        let prediction = linear_layer.forward(batch);
        let _ = loss.forward(prediction, labels);
        let t1 = loss.backward();
        let _ = linear_layer.backward(t1, 0.05);
    }

    // Ensure loss is below 0.001
    assert!(loss.forward(linear_layer.forward(batch), labels) < 1e-3);
}

#[test]
fn logistic_classifier_small() {
    let batch = Batch::<4, f64, 2> ([
        Tensor::<f64, 2> ([0.0, 1.0]),
        Tensor::<f64, 2> ([1.0, 1.0]),
        Tensor::<f64, 2> ([2.0, 1.0]),
        Tensor::<f64, 2> ([3.0, 1.0]),
    ]);

    let labels = Batch::<4, f64, 2> ([
        Tensor::<f64, 2> ([1.0, 0.0]),
        Tensor::<f64, 2> ([1.0, 0.0]),
        Tensor::<f64, 2> ([0.0, 1.0]),
        Tensor::<f64, 2> ([0.0, 1.0]),
    ]);

    // Initialize linear layer
    let mut linear_layer = Linear::<4, f64, 2, 2>::new();

    // Initialize activation function
    let mut activation = Softmax::<4, f64, 2>::new();

    // Initialize loss function
    let mut loss = CrossEntropyLoss::<4, f64, 2>::new();

    for _ in 0..60 {
        let prediction = linear_layer.forward(batch);
        let p = activation.forward(prediction);
        let _ = loss.forward(p, labels);
        let t1 = loss.backward();
        let t2 = activation.backward(t1);
        let _ = linear_layer.backward(t2, 0.5);
    }

    // Ensure loss is below 0.001
    assert!(loss.forward(activation.forward(linear_layer.forward(batch)), labels) < 5e-2);
}

#[test]
fn logistic_classifier_large() {
    let batch = Batch::<8, f64, 2> ([
        Tensor::<f64, 2> ([0.0, 1.0]),
        Tensor::<f64, 2> ([1.0, 1.0]),
        Tensor::<f64, 2> ([2.0, 1.0]),
        Tensor::<f64, 2> ([3.0, 1.0]),
        Tensor::<f64, 2> ([4.0, 1.0]),
        Tensor::<f64, 2> ([5.0, 1.0]),
        Tensor::<f64, 2> ([6.0, 1.0]),
        Tensor::<f64, 2> ([7.0, 1.0]),
    ]);

    let labels = Batch::<8, f64, 4> ([
        Tensor::<f64, 4> ([1.0, 0.0, 0.0, 0.0]),
        Tensor::<f64, 4> ([1.0, 0.0, 0.0, 0.0]),
        Tensor::<f64, 4> ([0.0, 1.0, 0.0, 0.0]),
        Tensor::<f64, 4> ([0.0, 1.0, 0.0, 0.0]),
        Tensor::<f64, 4> ([0.0, 0.0, 1.0, 0.0]),
        Tensor::<f64, 4> ([0.0, 0.0, 1.0, 0.0]),
        Tensor::<f64, 4> ([0.0, 0.0, 0.0, 1.0]),
        Tensor::<f64, 4> ([0.0, 0.0, 0.0, 1.0]),
    ]);

    // Initialize linear layer
    let mut linear_layer = Linear::<8, f64, 2, 4>::new();

    // Initialize activation function
    let mut activation = Softmax::<8, f64, 4>::new();

    // Initialize loss function
    let mut loss = CrossEntropyLoss::<8, f64, 4>::new();

    for _ in 0..5000 {
        let prediction = linear_layer.forward(batch);
        let p = activation.forward(prediction);
        let _ = loss.forward(p, labels);
        let t1 = loss.backward();
        let t2 = activation.backward(t1);
        let _ = linear_layer.backward(t2, 0.12);
    }

    // Ensure loss is below 0.001
    assert!(loss.forward(activation.forward(linear_layer.forward(batch)), labels) < 5e-2);
}