pub mod mean_squared;

use crate::model::{DATA, fXX};

pub struct LossResult {
    pub loss: fXX,
    pub gradient: DATA
}

#[typetag::serde]
pub trait Loss {
    fn loss(&mut self, predicted: DATA, actual: DATA) -> LossResult;
}