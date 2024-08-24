use ndarray_stats::DeviationExt;
use serde::{Deserialize, Serialize};
use crate::loss::{Loss, LossResult};
use crate::model::DATA;

#[derive(Serialize, Deserialize)]
pub struct MeanSquared {}

impl MeanSquared {
    pub fn new() -> MeanSquared {
        MeanSquared {}
    }
}

#[typetag::serde(name="Mean Squared Loss")]
impl Loss for MeanSquared {
    fn loss(&mut self, predicted: DATA, actual: DATA) -> LossResult {
        LossResult {
            loss: (&predicted - &actual).mapv(|e| e.powi(2)).mean().unwrap(),
            gradient: (&predicted - &actual) * 2.0
        }
    }
}