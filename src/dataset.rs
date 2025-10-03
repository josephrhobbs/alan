//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Dataset abstraction.

use rand::{
    seq::SliceRandom,
    rng,
};

use crate::{
    Numeric,
    tensor::{
        Batch,
        Tensor,
    },
};

#[derive(Clone, Debug)]
/// Dataset for training or testing.
pub struct Dataset<const B: usize, T: Numeric, const N: usize, const M: usize> {
    /// Data.
    data: Vec<Tensor<T, N>>,

    /// Labels.
    labels: Vec<Tensor<T, M>>,

    /// Indices from which to yield.
    shuffle: Vec<usize>,

    /// Number of batches yielded so far.
    batch: usize,
}

impl<const B: usize, T: Numeric, const N: usize, const M: usize> Dataset<B, T, N, M> {
    /// Construct a new dataset from a given list of data and their labels.
    /// 
    /// This function returns `Option<Dataset<B, T, N, M>>`.  If `data` and `labels` are
    /// different lengths, the function returns `None`.
    pub fn new(data: Vec<Tensor<T, N>>, labels: Vec<Tensor<T, M>>) -> Option<Self> {
        if data.len() != labels.len() {
            None
        } else if data.len() % B != 0 {
            None
        } else {
            let mut rng = rng();
            let mut shuffle: Vec<usize> = (0..B).collect();
            shuffle.shuffle(&mut rng);

            Some (Self {
                data,
                labels,
                shuffle,
                batch: 0,
            })
        }
    }

    /// Yield the next batch from this dataset, if it is available.
    pub fn next(&mut self) -> Option<(Batch<B, T, N>, Batch<B, T, M>)> {
        // Do we have enough for a batch?
        if B*(self.batch+1) > self.data.len() {
            None
        } else {
            let mut thisdata = [Tensor::zero(); B];
            let mut thislabel = [Tensor::zero(); B];
            for (e, i) in (B*self.batch..B*(self.batch+1)).enumerate() {
                thisdata[e] = self.data[self.shuffle[i]];
                thislabel[e] = self.labels[self.shuffle[i]];
            }
            self.batch += 1;

            Some ((Batch (thisdata), Batch (thislabel)))
        }
    } 

    /// Refresh this dataset.
    pub fn refresh(&mut self) {
        // Create new random ordering
        let mut rng = rng();
        let mut shuffle: Vec<usize> = (0..B).collect();
        shuffle.shuffle(&mut rng);
        self.shuffle = shuffle;

        // Reset batch number
        self.batch = 0;
    }
}
