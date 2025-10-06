//! ALAN
//! Copyright (c) 2025 J. Hobbs
//!
//! Numeric data type abstraction.

mod fixed16;

use std::{
    fmt::Debug,
    ops::{
        Add,
        Div,
        Mul,
        Neg,
        Sub,
    },
};

use rand;

pub use fixed16::x16;

/// Numeric data type.
/// 
/// A data type `T` can be `Numeric` if it defines the following.
/// - Addition over itself: `T + T -> T`
/// - Subtraction
/// TODO
pub trait Numeric: Clone + Copy + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Div<Output = Self> + Neg<Output = Self> + PartialOrd + Debug {
    fn zero() -> Self;

    fn one() -> Self;

    fn exp(self) -> Self;

    fn log(self) -> Self;

    fn random() -> Self;

    fn tiny() -> Self;

    fn neginf() -> Self;
}

impl Numeric for f32 { 
    fn zero() -> Self {
        0.0f32
    }

    fn one() -> Self {
        1.0f32
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn log(self) -> Self {
        self.ln()
    }

    fn random() -> Self {
        rand::random()
    }

    fn tiny() -> Self {
        1e-4
    }

    fn neginf() -> Self {
        f32::NEG_INFINITY
    }
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

    fn random() -> Self {
        rand::random()
    }

    fn tiny() -> Self {
        1e-4
    }
    
    fn neginf() -> Self {
        f64::NEG_INFINITY
    }
}

impl Numeric for x16 { 
    fn zero() -> Self {
        x16::zero()
    }

    fn one() -> Self {
        x16::one()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn log(self) -> Self {
        self.log()
    }

    fn random() -> Self {
        Self::random()
    }

    fn tiny() -> Self {
        Self::tiny()
    }
    
    fn neginf() -> Self {
        Self::neginf()
    }
}
