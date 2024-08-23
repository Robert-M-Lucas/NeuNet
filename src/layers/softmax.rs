use ndarray::{array, Array};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use typetag::serde;

use crate::layers::{DATA, Layer};
use crate::model::DataElement;

#[derive(Serialize, Deserialize)]
pub struct SoftmaxActivationConfig {
    size: Vec<usize>
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
        Self::new_multi_dim(vec![size])
    }

    pub fn new_multi_dim(size: Vec<usize>) -> SoftmaxActivator {
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
        self.config.size.clone()
    }

    fn output_shape(&self) -> Vec<usize> {
        self.config.size.clone()
    }

    fn forward_actual(&mut self, val: DATA, save_context: bool) -> DATA {
        let max = val.max().unwrap();

        let exp = val.mapv(|e| e - max);
        let exp_sum = exp.sum();

        let output = if exp_sum == 0.0 {
            Array::from_elem(val.shape(), 1.0 / val.shape().iter().fold(1, |a, b| a * b) as DataElement)
        }
        else {
            exp.mapv(|e| e / exp_sum)
        };

        if save_context {
            self.back_context = Some(val);
        }

        output
    }

    fn data_bin(&self) -> Vec<Vec<u8>> {
        vec![]
    }

    fn load_data(&mut self, data: Vec<Vec<u8>>) {}
}
