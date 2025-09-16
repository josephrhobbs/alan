//! ALAN Testbench

use alan::{
    tensor::{
        Batch,
        Tensor,
    },
    network::{
        Layer,
        Linear,
    },
    optim::{
        Loss,
        SSELoss,
    },
};

#[test]
fn linear_regression() {
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
    let mut loss = SSELoss::<4, f64, 2>::new();

    for i in 0..25 {
        let prediction = linear_layer.forward(batch);

        let l = loss.forward(prediction, labels);
        println!("Epoch {} | SSE: {:.4}", i, l);

        let t1 = loss.backward();
        let _ = linear_layer.backward(t1, 0.05);
    }

    // Ensure loss is below 0.005
    assert!(loss.forward(linear_layer.forward(batch), labels) < 5e-3);
}