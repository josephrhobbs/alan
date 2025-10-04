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

    /// Layer bias.
    pub bias: [[T; K]; K],

    /// Size of the kernel, as `T`.
    ksq_as_t: T,
}

impl<const B: usize, T: Numeric, const W: usize, const H: usize, const N: usize, const X: usize, const Y: usize, const M: usize, const K: usize> Layer<B, T, N, M> for Convolution<B, T, W, H, N, X, Y, M, K> {
    /// Construct a new layer, setting all parameters randomly.
    fn new() -> Self {
        // Initialize kernel and bias randomly
        let mut kernel = [[T::zero(); K]; K];
        let mut bias   = [[T::zero(); K]; K];
        let mut ksq_as_t = T::zero();
        for i in 0..K {
            for j in 0..K {
                kernel[i][j] = T::random();
                bias[i][j] = T::random();
                ksq_as_t = ksq_as_t + T::one();
            }
        }
    
        Self {
            input: Batch::<B, T, N>::zero(),
            kernel,
            bias,
            ksq_as_t,
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
                            result[b][j*X+i] = result[b][j*X+i] + self.kernel[ky][kx] * input[(j+ky)*W+(i+kx)] + self.bias[ky][kx] * T::one() / self.ksq_as_t;
                        }
                    }
                }
            }
        }

        result
    }

    fn backward(&mut self, gradients: &Batch<B, T, M>, lr: T) -> Batch<B, T, N> {
        let mut result = Batch::zero();

        // Compute gradients of input values
        for b in 0..B {
            let gradient = gradients[b];

            for i in 0..W {
                for j in 0..H {
                    for kx in 0..K {
                        for ky in 0..K {
                            // NOTE we add `K` here and subtract it later to prevent
                            // integer underflow with type `usize`
                            let gy = j + K - ky;
                            let gx = i + K - kx;
                            let g = if K <= gx && gx < X+K && K <= gy && gy < Y+K {
                                gradient[(gy-K)*X+(gx-K)]
                            } else {
                                T::zero()
                            };
                            result[b][j*W+i] = result[b][j*W+i] + self.kernel[ky][kx] * g;
                        }
                    }
                }
            }
        }

        // Update parameters
        for b in 0..B {
            let input = self.input[b];
            let gradient = gradients[b];

            for kx in 0..K {
                for ky in 0..K {
                    for i in 0..X {
                        for j in 0..Y {
                            self.kernel[ky][kx] = self.kernel[ky][kx] - lr * input[(j+ky)*W+(i+kx)] * gradient[j*X+i];

                            // Derivative of output gradient is unity wrt bias
                            self.bias[ky][kx] = self.bias[ky][kx] - lr * gradient[j*X+i];
                        }
                    }
                }
            }
        }

        result
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
    conv.bias = [
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
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

    // Compute input gradients and expected result
    let gradients = Batch::<1, f64, 9> ([Tensor::<f64, 9> ([
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    ])]);
    let input_gradients = conv.backward(&gradients, 1.0);
    let expected = Batch::<1, f64, 25> ([Tensor::<f64, 25> ([
        -1.0, -1.0,  0.0,  1.0,  1.0,
        -3.0, -3.0,  0.0,  3.0,  3.0,
        -4.0, -4.0,  0.0,  4.0,  4.0,
        -3.0, -3.0,  0.0,  3.0,  3.0,
        -1.0, -1.0,  0.0,  1.0,  1.0,
    ])]);
    assert_eq!(expected, input_gradients);

    // Check kernel
    let expected_kernel = [
        [-4.0, -3.0, -2.0],
        [-5.0, -3.0, -1.0],
        [-4.0, -3.0, -2.0],
    ];
    assert_eq!(conv.kernel, expected_kernel);

    // Check bias
    let expected_bias = [
        [-9.0, -9.0, -9.0],
        [-9.0, -9.0, -9.0],
        [-9.0, -9.0, -9.0],
    ];
    assert_eq!(conv.bias, expected_bias);
}
