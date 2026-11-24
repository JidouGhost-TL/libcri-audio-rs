//! Portable `f32` math for `no_std` builds (no `libm` dependency).

/// Pi as `f32`.
#[allow(clippy::approx_constant)]
pub const PI: f32 = 3.141_592_7;
#[allow(clippy::approx_constant)]
const HALF_PI: f32 = 1.570_796_4;
const TWO_PI: f32 = 6.283_185_5;
#[allow(clippy::approx_constant)]
const LN_2: f32 = 0.693_147_2;

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

/// Cosine via quadrant reduction + degree-12 Horner polynomial (full `f32` precision, core-only).
pub fn cos_f32(x: f32) -> f32 {
    let mut r = x;
    let k = (r / TWO_PI) as i32;
    r -= k as f32 * TWO_PI;
    while r > PI {
        r -= TWO_PI;
    }
    while r < -PI {
        r += TWO_PI;
    }
    let sign = if r > HALF_PI {
        r = PI - r;
        -1.0f32
    } else if r < -HALF_PI {
        r = -PI - r;
        -1.0f32
    } else {
        1.0f32
    };
    let x2 = r * r;
    let p = 1.0
        + x2 * (-0.5
            + x2 * (1.0 / 24.0
                + x2 * (-1.0 / 720.0
                    + x2 * (1.0 / 40_320.0
                        + x2 * (-1.0 / 3_628_800.0 + x2 * (1.0 / 479_001_600.0))))));
    sign * p
}

/// Sine via quadrant-reduced cosine.
pub fn sin_f32(x: f32) -> f32 {
    cos_f32(x - HALF_PI)
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

/// `2^x` for real exponents (deterministic, core-only).
pub fn pow2_f32(x: f32) -> f32 {
    let n = if x >= 0.0 {
        x as i32
    } else {
        let t = x as i32;
        if (t as f32) > x {
            t - 1
        } else {
            t
        }
    };
    let f = x - n as f32; // in [0.0, 1.0)
    let u = f * LN_2;
    // exp(u) via degree-7 Horner polynomial on [0, ln 2)
    let exp_u = 1.0
        + u * (1.0
            + u * (0.5
                + u * (1.0 / 6.0
                    + u * (1.0 / 24.0
                        + u * (1.0 / 120.0 + u * (1.0 / 720.0 + u * (1.0 / 5040.0)))))));
    pow2_i32(n) * exp_u
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trig_and_pow2_accuracy() {
        assert!((cos_f32(0.0) - 1.0).abs() < 1e-6);
        assert!((cos_f32(PI) + 1.0).abs() < 1e-6);
        assert!((cos_f32(-PI) + 1.0).abs() < 1e-6);
        assert!((sin_f32(HALF_PI) - 1.0).abs() < 1e-6);
        assert!((sin_f32(-HALF_PI) + 1.0).abs() < 1e-6);
        assert!((sqrt_f32(4.0) - 2.0).abs() < 1e-6);
        assert!((sqrt_f32(2.0) - core::f32::consts::SQRT_2).abs() < 1e-6);
        assert!((pow2_f32(0.5) - core::f32::consts::SQRT_2).abs() < 1e-5);
        assert!((pow2_f32(-1.0) - 0.5).abs() < 1e-6);
        assert!((pow2_f32(3.0) - 8.0).abs() < 1e-5);
    }
}
