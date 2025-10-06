//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Image processing testbench.

use image::{
    GrayImage,
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

fn load_image(path: &str) -> Tensor<f32, 65536> {
    // Open image and convert to grayscale
    let image: GrayImage = open(path).unwrap().to_luma8();
    
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
fn train_image_classifier() {
    let fire_hydrant = load_image("images/hydrant.jpg");
    let mit          = load_image("images/mit.jpg");
    let mut dataset: Dataset<2, f32, 65536, 2> = Dataset::new(
        vec![
            fire_hydrant,
            mit,
        ],
        vec![
            Tensor::<f32, 2> ([1.0, 0.0]),
            Tensor::<f32, 2> ([0.0, 1.0]),
        ]
    ).unwrap();

    // Initialize classifier
    let mut classifier: ImageClassifier<2, f32, 2> = ImageClassifier::new();

    // Train classifier
    let h = Hyperparameters {epochs: 50, lr: 1e-4};
    classifier.train(&mut dataset, h);

    // Compute class probabilities
    let inference = classifier.eval(&Batch::<2, f32, 65536> ([fire_hydrant, mit]));

    assert!(1.0 - inference[0][0] < 1e-3);
}
