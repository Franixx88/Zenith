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

/// Per-node confidence shared by pruning and reductions. Correction is in cp,
/// not the fixed-point units used internally by correction-history tables.
#[derive(Clone, Copy)]
pub(super) struct Selectivity {
    pub depth: i16,
    pub tt_pv: bool,
    pub cut_node: bool,
    pub improving: bool,
    pub correction: i32,
}
impl Selectivity {
    pub fn rfp_margin(self, base: i32) -> i32 {
        base * i32::from(self.depth) + 70 - i32::from(self.improving) * base / 2
            + 2 * self.correction
    }
    pub fn null_reduction(self, base: i16, eval_margin: i32) -> i16 {
        (base + self.depth / 4 + (eval_margin.max(0) / 160).min(3) as i16
            - i16::from(self.tt_pv)
            - (self.correction / 64).min(2) as i16)
            .clamp(2, (self.depth - 1).max(2))
    }
    pub fn lmr(self, base: i16, history: i32) -> i16 {
        (base + i16::from(self.cut_node)
            - i16::from(self.improving)
            - i16::from(self.tt_pv)
            - (history / 4000) as i16
            - (self.correction / 80).min(2) as i16)
            .clamp(0, (self.depth - 2).max(0))
    }
    pub fn quiet_limit(self, lmr_depth: i16) -> usize {
        (3 + usize::from(lmr_depth.max(0) as u16).pow(2)) * if self.improving { 2 } else { 1 }
    }
    pub fn futility_margin(self, base: i32, lmr_depth: i16) -> i32 {
        100 + base * i32::from(lmr_depth.max(0) + 1)
            + 2 * self.correction
            + 80 * i32::from(self.tt_pv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uncertainty_and_pv_protect_depth_while_eval_surplus_strengthens_null() {
        let plain = Selectivity {
            depth: 10,
            tt_pv: false,
            cut_node: true,
            improving: false,
            correction: 0,
        };
        let uncertain = Selectivity {
            correction: 128,
            ..plain
        };
        let pv = Selectivity {
            tt_pv: true,
            ..plain
        };
        assert!(uncertain.rfp_margin(120) > plain.rfp_margin(120));
        assert!(uncertain.futility_margin(120, 3) > plain.futility_margin(120, 3));
        assert!(uncertain.lmr(4, 0) < plain.lmr(4, 0));
        assert!(pv.lmr(4, 0) < plain.lmr(4, 0));
        assert!(plain.null_reduction(3, 640) > plain.null_reduction(3, 0));
        assert!(uncertain.null_reduction(3, 0) < plain.null_reduction(3, 0));
        assert!(plain.lmr(4, 8000) < plain.lmr(4, -8000));
        for depth in 3..=120 {
            let p = Selectivity { depth, ..uncertain };
            assert!((0..=depth - 2).contains(&p.lmr(120, -16000)));
            assert!((2..depth).contains(&p.null_reduction(3, 30_000)));
        }
    }
}
