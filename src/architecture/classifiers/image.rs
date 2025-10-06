//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Image classifier.

use crate::{
    architecture::Architecture,
    network::{
        activation::Softmax,
        layer::{
            AvgPool,
            Convolution,
            Linear,
        },
        Layer,
    },
    Numeric,
    optim::loss::CrossEntropyLoss,
    tensors::Batch,
};

/// Multi-class image classifier.
/// 
/// This image classifier accepts 256x256 grayscale images
/// and classifies them into `C` classes.
pub struct ImageClassifier<const B: usize, T: Numeric, const C: usize> {
    /// Convolution layer 1
    conv1: Convolution<B, T, 256, 256, 65536, 250, 250, 62500, 7>,

    /// Average pooling 1
    avgpool1: AvgPool<B, T, 250, 250, 62500, 50, 50, 2500, 5>,

    /// Convolution layer 2
    conv2: Convolution<B, T, 50, 50, 2500, 48, 48, 2304, 3>,

    /// Average pooling 2
    avgpool2: AvgPool<B, T, 48, 48, 2304, 24, 24, 576, 2>,

    /// Fully connected layer
    fc1: Linear<B, T, 576, C>,
}

/// Training and inference implementation for this image classifier.
impl<const B: usize, T: Numeric, const C: usize> Architecture<B, T, 65536, C> for ImageClassifier<B, T, C> {
    /// Mean squared error loss.
    type LossFunction = CrossEntropyLoss<B, T, C>;

    /// Softmax activation head.
    type Activation = Softmax<B, T, C>;

    /// Construct this regressor.
    fn new() -> Self {
        Self {
            conv1: Convolution::new(),
            avgpool1: AvgPool::new(),
            conv2: Convolution::new(),
            avgpool2: AvgPool::new(),
            fc1: Linear::new(),
        }
    }

    /// Compute the forward pass of this regressor.
    fn forward(&mut self, batch: &Batch<B, T, 65536>) -> Batch<B, T, C> {
        let fmap1 = self.avgpool1.forward(&self.conv1.forward(batch));
        let fmap2 = self.avgpool2.forward(&self.conv2.forward(&fmap1));
        
        let mut out = T::zero();
        for i in 0..fmap1.0[0].0.len() {
            out = out + fmap1.0[0].0[i];
        }

        self.fc1.forward(&fmap2)
    }

    /// Compute the backward pass of this regressor.
    fn backward(&mut self, gradients: &Batch<B, T, C>, lr: T) {
        let gmap2 = self.fc1.backward(gradients, lr);
        let gmap1 = self.conv2.backward(&self.avgpool2.backward(&gmap2, lr), lr);
        let _     = self.conv1.backward(&self.avgpool1.backward(&gmap1, lr), lr);
    }
}
