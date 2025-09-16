//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Numeric data type abstraction.

use std::ops::{
    Add,
    Div,
    Mul,
    Neg,
    Sub,
};

/// Numeric data type.
/// 
/// A data type `T` can be `Numeric` if it defines the following.
/// - Addition over itself: `T + T -> T`
/// - Subtraction
pub trait Numeric: Clone + Copy + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Div<Output = Self> + Neg<Output = Self> + PartialOrd {
    fn zero() -> Self;

    fn one() -> Self;

    fn exp(self) -> Self;

    fn log(self) -> Self;
}

impl Numeric for f64 { 
    fn zero() -> Self {
        0.0f64
    }

    fn one() -> Self {
        1.0f64
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn log(self) -> Self {
        self.ln()
    }
}