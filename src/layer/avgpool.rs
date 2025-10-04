//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Average pooling network layer.

use crate::{
    layer::Layer,
    Numeric,
    tensor::Batch,
};

/// Average pooling network layer.
/// 
/// An average pooling layer `AvgPool<B, T, W, H, N, X, Y, M, K>` maps `Tensor`s of size `N = W*H` and type `T`
/// to `Tensor`s of size `M = X * Y` and identical type using a square average pool of size `K`.  `Tensor`s
/// are evaluated in `Batch`es to improve performance.
pub struct AvgPool<const B: usize, T: Numeric, const W: usize, const H: usize, const N: usize, const X: usize, const Y: usize, const M: usize, const K: usize> {
    /// Last network layer input.
    input: Batch<B, T, N>,

    /// Layer kernel.
    pub kernel: [[T; K]; K],
}

impl<const B: usize, T: Numeric, const W: usize, const H: usize, const N: usize, const X: usize, const Y: usize, const M: usize, const K: usize> Layer<B, T, N, M> for AvgPool<B, T, W, H, N, X, Y, M, K> {
    /// Construct a new layer.
    fn new() -> Self {
        let mut ksq_as_t = T::zero();
        for _ in 0..K {
            for _ in 0..K {
                ksq_as_t = ksq_as_t + T::one();
            }
        }

        Self {
            input: Batch::<B, T, N>::zero(),
            kernel: [[T::one() / ksq_as_t; K]; K],
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
                            result[b][j*X+i] = result[b][j*X+i] + self.kernel[ky][kx] * input[(j*K+ky)*W+(i*K+kx)];
                        }
                    }
                }
            }
        }

        result
    }

    fn backward(&mut self, gradients: &Batch<B, T, M>, _lr: T) -> Batch<B, T, N> {
        let mut result = Batch::zero();

        // Compute gradients of input values
        for b in 0..B {
            let gradient = gradients[b];

            for i in 0..W {
                for j in 0..H {
                    for kx in 0..K {
                        for ky in 0..K {
                            // NOTE we add `K*K` here and subtract it later to prevent
                            // integer underflow with type `usize`
                            let gy = j + K*K - K*ky;
                            let gx = i + K*K - K*kx;
                            let g = if K*K <= gx && gx < X+K*K && K*K <= gy && gy < Y+K*K {
                                gradient[(gy-K*K)*X+(gx-K*K)]
                            } else {
                                T::zero()
                            };
                            result[b][j*W+i] = result[b][j*W+i] + self.kernel[ky][kx] * g;
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
fn test_avgpool_layer() {
    let mut avgpool = AvgPool::<1, f64, 4, 4, 16, 2, 2, 4, 2>::new();

    // Input image
    let image = Batch::<1, f64, 16> ([Tensor::<f64, 16> ([
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        5.0, 6.0, 7.0, 8.0,
        1.0, 2.0, 3.0, 4.0,
    ])]);

    // Compute result and expected result
    let result = avgpool.forward(&image);
    let expected = Batch::<1, f64, 4> ([Tensor::<f64, 4> ([
        3.50, 5.50,
        3.50, 5.50,
    ])]);
    assert_eq!(result, expected);

    // Compute input gradients and expected result
    let gradients = Batch::<1, f64, 4> ([Tensor::<f64, 4> ([
        1.0, 1.0,
        1.0, 1.0,
    ])]);
    let input_gradients = avgpool.backward(&gradients, 1.0);
    let expected = Batch::<1, f64, 16> ([Tensor::<f64, 16> ([
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
    ])]);
    assert_eq!(expected, input_gradients);
}
