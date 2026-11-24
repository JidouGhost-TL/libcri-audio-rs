//! Portable `f32` math for `no_std` builds (no `libm` dependency).

/// Pi as `f32`.
#[allow(clippy::approx_constant)]
pub const PI: f32 = 3.141_592_7;
const TWO_PI: f32 = 6.283_185_5;

/// Square root via Newton iteration (deterministic, core-only).
pub fn sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = if x >= 1.0 { x } else { 1.0 };
    for _ in 0..24 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// Cosine via range reduction + Taylor series (deterministic, core-only).
pub fn cos_f32(x: f32) -> f32 {
    // Reduce to [-pi, pi].
    let mut r = x;
    while r > PI {
        r -= TWO_PI;
    }
    while r < -PI {
        r += TWO_PI;
    }
    let x2 = r * r;
    // cos(r) = 1 - x2/2! + x4/4! - x6/6! + x8/8! - x10/10! + x12/12!
    let mut term = 1.0f32;
    let mut sum = 1.0f32;
    let mut n = 1u32;
    while n <= 6 {
        term *= -x2 / (((2 * n - 1) * (2 * n)) as f32);
        sum += term;
        n += 1;
    }
    sum
}

/// Sine via range reduction + Taylor series.
pub fn sin_f32(x: f32) -> f32 {
    cos_f32(x - PI * 0.5)
}

/// `2^n` for small integer exponents.
pub fn pow2_i32(n: i32) -> f32 {
    let mut v = 1.0f32;
    if n >= 0 {
        for _ in 0..n {
            v *= 2.0;
        }
    } else {
        for _ in 0..(-n) {
            v *= 0.5;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trig_accuracy() {
        assert!((cos_f32(0.0) - 1.0).abs() < 1e-6);
        assert!((cos_f32(PI) + 1.0).abs() < 1e-3);
        assert!((sin_f32(PI * 0.5) - 1.0).abs() < 1e-4);
        assert!((sqrt_f32(4.0) - 2.0).abs() < 1e-5);
        assert!((sqrt_f32(2.0) - core::f32::consts::SQRT_2).abs() < 1e-5);
    }
}
