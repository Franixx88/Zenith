//! Small, independently testable search controls. Baseline values are retained
//! until a held-out match supports a change; these are not SPSA-tuned constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchParameters {
    pub rfp_margin: i32,
    pub futility_margin: i32,
    pub nmp_base_reduction: i16,
    /// Hundredths: 300 means ln(depth)*ln(move_number)/3.0.
    pub lmr_divisor: u16,
}
impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            rfp_margin: 120,
            futility_margin: 120,
            nmp_base_reduction: 3,
            lmr_divisor: 300,
        }
    }
}
impl SearchParameters {
    pub fn validate(self) -> Result<Self, String> {
        if !(40..=300).contains(&self.rfp_margin)
            || !(40..=300).contains(&self.futility_margin)
            || !(2..=5).contains(&self.nmp_base_reduction)
            || !(150..=600).contains(&self.lmr_divisor)
        {
            return Err(
                "RFP/Futility Margin: 40..300; NMP Base Reduction: 2..5; LMR Divisor: 150..600"
                    .into(),
            );
        }
        Ok(self)
    }
    pub(super) fn reductions(self) -> Vec<i16> {
        (0..128 * 256)
            .map(|index| {
                let depth = (index / 256) as f64;
                let number = (index % 256) as f64;
                (1.0 + depth.max(1.0).ln() * number.max(1.0).ln()
                    / (f64::from(self.lmr_divisor) / 100.0)) as i16
            })
            .collect()
    }
}
