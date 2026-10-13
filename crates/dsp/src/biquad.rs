//! Second-order IIR filters designed from the RBJ "Audio EQ Cookbook" formulas, plus first-order
//! low/high-pass sections for 6 dB/oct slopes.
//!
//! The design formulas, transposed-direct-form-II state and magnitude response live in
//! `craft-audio-dsp::biquad` (the shared Craft DSP library); this module keeps SoundCraft's
//! public surface: the serde-serializable [`FilterType`], the ±48 dB gain clamp, coefficient
//! access for redraws, and the NaN-guarded sample path.

use craft_audio_dsp::biquad as shared;

use crate::util::sane_sr;

/// Filter response shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    Peak,
    LowShelf,
    HighShelf,
    AllPass,
    /// First-order (6 dB/oct) low-pass; `q` and `gain_db` are ignored.
    LowPass1,
    /// First-order (6 dB/oct) high-pass; `q` and `gain_db` are ignored.
    HighPass1,
}

impl From<FilterType> for shared::FilterType {
    fn from(kind: FilterType) -> Self {
        match kind {
            FilterType::LowPass => shared::FilterType::LowPass,
            FilterType::HighPass => shared::FilterType::HighPass,
            FilterType::BandPass => shared::FilterType::BandPass,
            FilterType::Notch => shared::FilterType::Notch,
            FilterType::Peak => shared::FilterType::Peak,
            FilterType::LowShelf => shared::FilterType::LowShelf,
            FilterType::HighShelf => shared::FilterType::HighShelf,
            FilterType::AllPass => shared::FilterType::AllPass,
            FilterType::LowPass1 => shared::FilterType::LowPass1,
            FilterType::HighPass1 => shared::FilterType::HighPass1,
        }
    }
}

/// Normalized biquad coefficients (`a0 = 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Default for Coeffs {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Coeffs {
    /// Passes the signal unchanged.
    pub const IDENTITY: Coeffs = Coeffs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    fn from_shared(c: shared::Coeffs) -> Self {
        Self { b0: c.b0, b1: c.b1, b2: c.b2, a1: c.a1, a2: c.a2 }
    }

    fn to_shared(self) -> shared::Coeffs {
        shared::Coeffs { b0: self.b0, b1: self.b1, b2: self.b2, a1: self.a1, a2: self.a2 }
    }

    /// Designs a filter. All inputs are sanitized: frequency is clamped to 1 Hz .. 0.49·sr,
    /// Q to 0.025 .. 100 and gain to ±48 dB.
    pub fn design(kind: FilterType, freq: f32, q: f32, gain_db: f32, sample_rate: f32) -> Coeffs {
        let _ = sane_sr(sample_rate); // documented clamp; the shared designer re-sanitizes
        let g = if gain_db.is_finite() { gain_db.clamp(-48.0, 48.0) } else { 0.0 };
        let c = Coeffs::from_shared(shared::Coeffs::design(kind.into(), freq, q, g, sample_rate));
        if [c.b0, c.b1, c.b2, c.a1, c.a2].iter().all(|v| v.is_finite()) { c } else { Coeffs::IDENTITY }
    }

    /// Magnitude response in dB at `freq` Hz.
    pub fn magnitude_db(&self, freq: f32, sample_rate: f32) -> f32 {
        self.to_shared().magnitude_db(freq, sample_rate)
    }
}

/// A biquad section with its own state (transposed direct form II, in `craft-audio-dsp`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    inner: shared::Biquad,
}

impl Biquad {
    pub fn new(coeffs: Coeffs) -> Self {
        Self { inner: shared::Biquad::new(coeffs.to_shared()) }
    }

    /// Designs a new filter (see [`Coeffs::design`]).
    pub fn design(kind: FilterType, freq: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        Self::new(Coeffs::design(kind, freq, q, gain_db, sample_rate))
    }

    /// Re-designs the coefficients, keeping the state (for smooth sweeps).
    pub fn set(&mut self, kind: FilterType, freq: f32, q: f32, gain_db: f32, sample_rate: f32) {
        self.set_coeffs(Coeffs::design(kind, freq, q, gain_db, sample_rate));
    }

    /// Swaps in pre-designed coefficients, keeping the state.
    pub fn set_coeffs(&mut self, coeffs: Coeffs) {
        self.inner.update(coeffs.to_shared());
    }

    /// The current coefficients.
    pub fn coeffs(&self) -> Coeffs {
        Coeffs::from_shared(self.inner.coeffs)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.inner.process(x);
        if y.is_finite() {
            y
        } else {
            self.reset();
            0.0
        }
    }

    pub fn process_block(&mut self, buf: &mut [f32]) {
        for s in buf {
            *s = self.process_sample(*s);
        }
    }

    /// Magnitude response in dB at `freq` Hz, for drawing curves.
    pub fn magnitude_db(&self, freq: f32, sample_rate: f32) -> f32 {
        self.inner.coeffs.magnitude_db(freq, sample_rate)
    }
}

/// Butterworth Q values for cascaded sections making an `order`-th order filter (2, 4).
pub(crate) fn butterworth_qs(order: usize) -> &'static [f32] {
    match order {
        4 => &[0.541_196_1, 1.306_563],
        _ => &[std::f32::consts::FRAC_1_SQRT_2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_gain_at_center() {
        let c = Coeffs::design(FilterType::Peak, 1000.0, 1.0, 6.0, 48_000.0);
        assert!((c.magnitude_db(1000.0, 48_000.0) - 6.0).abs() < 0.01);
        assert!(c.magnitude_db(20.0, 48_000.0).abs() < 0.1);
    }

    #[test]
    fn lowpass_minus3_at_cutoff() {
        let c = Coeffs::design(FilterType::LowPass, 1000.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, 48_000.0);
        assert!((c.magnitude_db(1000.0, 48_000.0) + 3.01).abs() < 0.05);
        let c1 = Coeffs::design(FilterType::HighPass1, 1000.0, 0.0, 0.0, 48_000.0);
        assert!((c1.magnitude_db(1000.0, 48_000.0) + 3.01).abs() < 0.05);
    }

    #[test]
    fn hostile_inputs() {
        let c = Coeffs::design(FilterType::Peak, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0);
        assert!(c.b0.is_finite());
        let mut b = Biquad::new(c);
        assert!(b.process_sample(f32::NAN) == 0.0);
    }

    #[test]
    fn notch_reports_depth_not_flat_zero() {
        // The shared magnitude response must draw notches with their true depth.
        let c = Coeffs::design(FilterType::Notch, 1000.0, 8.0, 0.0, 48_000.0);
        assert!(c.magnitude_db(1000.0, 48_000.0) < -60.0);
    }

    #[test]
    fn set_coeffs_keeps_state_and_updates() {
        let mut b = Biquad::design(FilterType::LowPass, 100.0, 1.0, 0.0, 48_000.0);
        let y1 = b.process_sample(1.0);
        assert!(y1 > 0.0);
        b.set_coeffs(Coeffs::IDENTITY);
        assert_eq!(b.coeffs(), Coeffs::IDENTITY);
        // The filter state survived the swap: over silence the identity filter drains
        // z1 and z2, then outputs exact zeros.
        let r1 = b.process_sample(0.0);
        let r2 = b.process_sample(0.0);
        let r3 = b.process_sample(0.0);
        assert!(r1 != 0.0 || r2 != 0.0, "state was cleared by the coefficient swap");
        assert_eq!(r3, 0.0);
    }
}
