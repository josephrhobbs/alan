//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Image processing testbench.

use image::{
    GrayImage,
    ImageBuffer,
    open,
};

use alan::{
    models::{
        Architecture,
        classifiers::ImageClassifier,
    },
    optim::Hyperparameters,
    tensor::{
        Dataset,
        Batch,
        Tensor,
    },
};

fn load_fire_hydrant() -> Tensor<f32, 65536> {
    // Open fire hydrant image and convert to grayscale
    let image: GrayImage = open("images/hydrant.jpg").unwrap().to_luma8();
    
    // Convert image to tensor
    let array: [f32; 65536] = (
        *image.as_raw()
        .into_iter()
        .map(|x| *x as f32 / 255.0)
        .collect::<Vec<f32>>()
    ).try_into().unwrap();

    Tensor (array)
}

#[test]
fn train_classifier() {
    let tensor = load_fire_hydrant();
    let mut dataset: Dataset<1, f32, 65536, 2> = Dataset::new(vec![tensor], vec![Tensor::<f32, 2> ([1.0, 0.0])]).unwrap();

    // Initialize classifier
    let mut classifier: ImageClassifier<1, f32, 2> = ImageClassifier::new();

    // Train classifier
    let h = Hyperparameters {epochs: 25, lr: 0.005};
    classifier.train(&mut dataset, h);

    // Compute class probabilities
    let inference = classifier.forward(&Batch::<1, f32, 65536> ([tensor]));
    dbg!(inference);
}
