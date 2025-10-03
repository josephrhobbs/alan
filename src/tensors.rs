//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! Tensor implementation.

use std::ops::{
    Index,
    IndexMut,
};

use crate::Numeric;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tensor<T: Numeric, const N: usize> (pub [T; N]);

impl<T: Numeric, const N: usize> Tensor<T, N> {
    pub fn zero() -> Self {
        Self ([T::zero(); N])
    }
}

impl<T: Numeric, const N: usize> Index<usize> for Tensor<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Numeric, const N: usize> IndexMut<usize> for Tensor<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Batch<const B: usize, T: Numeric, const N: usize> (pub [Tensor<T, N>; B]);

impl<const B: usize, T: Numeric, const N: usize> Batch<B, T, N> {
    pub fn zero() -> Self {
        Self ([Tensor::<T, N>::zero(); B])
    }
}

impl<const B: usize, T: Numeric, const N: usize> Index<usize> for Batch<B, T, N> {
    type Output = Tensor<T, N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const B: usize, T: Numeric, const N: usize> IndexMut<usize> for Batch<B, T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
