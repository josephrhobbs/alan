//! ALAN
//! Copyright (c) 2025 J. Hobbs
//! 
//! 16-bit fixed-point format.

use std::{
    fmt,
    ops::{
        Add,
        Sub,
        Mul,
        Neg,
        Div,
    },
};

use rand::random;

/// Scale of 16-bit float.
const SCALE: i16 = 1_000;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, PartialOrd)]
/// Fixed-point value with 16 bits of precision.
pub struct x16 (i16);

impl x16 {
    /// Generate a random value between 0 and 1.
    pub fn random() -> Self {
        let r = SCALE as f32 * random::<f32>();
        Self (r as i16)
    }

    /// Zero.
    pub fn zero() -> Self {
        Self (0)
    }

    /// One.
    pub fn one() -> Self {
        Self (SCALE)
    }

    /// A tiny value.
    pub fn tiny() -> Self {
        Self (1)
    }

    /// Most negative possible value.
    pub fn neginf() -> Self {
        Self (i16::MIN)
    }

    /// Exponential.
    pub fn exp(self) -> Self {
        Self ((self.0 as f32).exp().powf(1.0 / SCALE as f32) as i16)
    }

    /// Natural logarithm.
    pub fn log(self) -> Self {
        Self (((self.0 as f32).ln() - (SCALE as f32).ln()) as i16)
    }
}

impl Add<x16> for x16 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        dbg!(self.0);
        dbg!(other.0);
        x16 (self.0 + other.0)
    }
}

impl Sub<x16> for x16 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self (self.0 - other.0)
    }
}

impl Mul<x16> for x16 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self (self.0 * (other.0 / SCALE))
    }
}

impl Div<x16> for x16 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self ((self.0 / other.0) * SCALE)
    }
}

impl Neg for x16 {
    type Output = Self;

    fn neg(self) -> Self {
        Self (-self.0) 
    }
}

impl fmt::Debug for x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0 as f32 / SCALE as f32)
    }
}

impl From<f32> for x16 {
    fn from(float: f32) -> x16 {
        Self ((float * SCALE as f32) as i16)
    }
}
