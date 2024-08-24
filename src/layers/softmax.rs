use ndarray::{array, Array};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use typetag::serde;

use crate::layers::{DATA, Layer};
use crate::model::fXX;

#[derive(Serialize, Deserialize)]
pub struct SoftmaxActivationConfig {
    size: usize
}

#[derive(Serialize, Deserialize)]
pub struct SoftmaxActivator {
    #[serde(flatten)]
    config: SoftmaxActivationConfig,
    #[serde(skip)]
    back_context: Option<DATA>
}

impl SoftmaxActivator {
    pub fn new(size: usize) -> SoftmaxActivator {
        SoftmaxActivator {
            config: SoftmaxActivationConfig {
                size
            },
            back_context: None,
        }
    }

    pub const fn name() -> &'static str {
        "Softmax Activation"
    }
}

#[typetag::serde(name = "Softmax Activation")]
impl Layer for SoftmaxActivator {
    fn name(&self) -> &'static str {
        Self::name()
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![self.config.size]
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.config.size]
    }

    fn forward_actual(&mut self, val: DATA, training: bool) -> DATA {
        let max = val.max().unwrap();

        let exp = val.mapv(|e| (e - max).exp());
        let exp_sum = exp.sum();

        let output = exp.mapv(|e| e / exp_sum);

        if training {
            self.back_context = Some(val);
        }

        output
    }

    fn backward_actual(&mut self, gradient: DATA, training_rate: fXX) -> DATA {
        todo!()
    }

    fn data_bin(&self) -> Vec<Vec<u8>> {
        vec![]
    }

    fn load_data(&mut self, data: Vec<Vec<u8>>) {}
}
