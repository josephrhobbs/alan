//! ALAN Testbench

use alan::{
    tensor::{
        Batch,
        Tensor,
    },
    network::{
        Layer,
        Linear,
        ReLU,
    },
};

#[test]
fn test_linear_layer_relu() {
    let t1 = Tensor::<f64, 3> ([3.0, 4.0, 5.0]);
    let batch = Batch::<1, f64, 3> ([t1]);

    let mut linear_layer = Linear::<1, f64, 3, 2> {
        input: Batch::<1, f64, 3>::zero(),
        parameters: [
            [-1.0, -2.0, 3.0],
            [4.0, -5.0, -6.0],
        ],
    };
    let mut relu = ReLU::<2>;

    let t2 = linear_layer.forward(batch);
    let t3 = relu.forward(t2);

    let t2b = relu.backward(t3, 0.01);
    let t1b = linear_layer.backward(t2, 0.01);

    dbg!(&linear_layer.parameters);
}