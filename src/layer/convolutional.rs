//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Convolutional network layer.

use crate::{
    layer::Layer,
    Numeric,
    tensor::Batch,
};

/// Convolutional network layer.
/// 
/// A convolutional layer `Convolution<B, T, W, H, N, X, Y, M, K>` maps `Tensor`s of size `N = W*H` and type `T`
/// to `Tensor`s of size `M = X * Y` and identical type using a square kernel of size `K`.  `Tensor`s
/// are evaluated in `Batch`es to improve performance.
pub struct Convolution<const B: usize, T: Numeric, const W: usize, const H: usize, const N: usize, const X: usize, const Y: usize, const M: usize, const K: usize> {
    /// Last network layer input.
    input: Batch<B, T, N>,
    
    /// Layer kernel.
    pub kernel: [[T; K]; K],
}

impl<const B: usize, T: Numeric, const W: usize, const H: usize, const N: usize, const X: usize, const Y: usize, const M: usize, const K: usize> Layer<B, T, N, M> for Convolution<B, T, W, H, N, X, Y, M, K> {
    /// Construct a new layer, setting all parameters randomly.
    fn new() -> Self {
        // Initialize kernel randomly
        let mut kernel = [[T::zero(); K]; K];
        for i in 0..K {
            for j in 0..K {
                kernel[i][j] = T::random();
            }
        }

        Self {
            input: Batch::<B, T, N>::zero(),
            kernel,
        }
    }
    
    fn forward(&mut self, batch: &Batch<B, T, N>) -> Batch<B, T, M> {
        self.input = *batch;

        let mut result = Batch::<B, T, M>::zero();

        for b in 0..B {
            let input = batch[b];

            for i in 0..X {
                for j in 0..Y {
                    for kx in 0..K {
                        for ky in 0..K {
                            println!("{} {} {} {}", i, j, kx, ky);
                            result[b][j*X+i] = result[b][j*X+i] + self.kernel[ky][kx] * input[(j+ky)*W+(i+kx)];
                        }
                    }
                }
            }
        }

        result
    }

    #[allow(unused_variables)]
    fn backward(&mut self, batch: &Batch<B, T, M>, lr: T) -> Batch<B, T, N> {
        todo!()
    }
}

#[cfg(test)]
use crate::tensor::Tensor;

#[test]
fn test_convolution_layer() {
    let mut conv = Convolution::<1, f64, 5, 5, 25, 3, 3, 9, 3>::new();
    conv.kernel = [
        [-1.0,  0.0,  1.0],
        [-2.0,  0.0,  2.0],
        [-1.0,  0.0,  1.0],
    ];

    // Input image
    let image = Batch::<1, f64, 25> ([Tensor::<f64, 25> ([
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
    ])]);

    // Compute result and expected result
    let result = conv.forward(&image);
    let expected = Batch::<1, f64, 9> ([Tensor::<f64, 9> ([
        4.0, 0.0, -4.0,
        4.0, 0.0, -4.0,
        4.0, 0.0, -4.0,
    ])]);

    assert_eq!(result, expected);
}
