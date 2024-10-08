use ndarray::{array, Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use typetag::serde;

use crate::layers::{DATA, Layer};
use crate::model::fXX;

#[derive(Serialize, Deserialize)]
pub struct ReluActivationConfig {
    size: Vec<usize>
}

#[derive(Serialize, Deserialize)]
pub struct ReluActivator {
    #[serde(flatten)]
    config: ReluActivationConfig,
    #[serde(skip)]
    back_context: Option<DATA>
}

impl ReluActivator {
    pub fn new(size: usize) -> ReluActivator {
        Self::new_multi_dim(vec![size])
    }

    pub fn new_multi_dim(size: Vec<usize>) -> ReluActivator {
        ReluActivator {
            config: ReluActivationConfig {
                size
            },
            back_context: None,
        }
    }

    pub const fn name() -> &'static str {
        "Relu Activation"
    }
}

#[typetag::serde(name = "Relu Activation")]
impl Layer for ReluActivator {
    fn name(&self) -> &'static str {
        Self::name()
    }

    fn input_shape(&self) -> Vec<usize> {
        self.config.size.clone()
    }

    fn output_shape(&self) -> Vec<usize> {
        self.config.size.clone()
    }

    fn forward_actual(&mut self, val: DATA, training: bool) -> DATA {
        let output = val.mapv(|e| e.max(0.0));

        if training {
            self.back_context = Some(val);
        }

        output
    }

    fn backward_actual(&mut self, gradient: DATA, training_rate: fXX) -> DATA {
        // TODO: What to do with 0? (no gradient)
        gradient * self.back_context.take().unwrap().mapv(|e| if e > 0.0 {
            1.0
        } else {
            0.0
        })
    }

    fn data_bin(&self) -> Vec<Vec<u8>> {
        vec![]
    }

    fn load_data(&mut self, data: Vec<Vec<u8>>) {}
}
