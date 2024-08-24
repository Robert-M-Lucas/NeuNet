use std::hint::black_box;
use ndarray::{array, Array, Array1, Array2, ArrayBase, Axis, OwnedRepr};
use serde::{Deserialize, Serialize};
use typetag::serde;

use crate::layers::{DATA, Layer};
use crate::model::fXX;

#[derive(Serialize, Deserialize)]
pub struct DropoutActivationConfig {
    size: usize,
    remove: usize
}

#[derive(Serialize, Deserialize)]
pub struct DropoutLayer {
    #[serde(flatten)]
    config: DropoutActivationConfig,
    #[serde(skip)]
    last_adjust: Option<Array1<fXX>>
}

impl DropoutLayer {
    pub fn new(size: usize, rate: fXX) -> DropoutLayer {
        Self::new_exact(size, (size as fXX * rate) as usize)
    }

    pub fn new_exact(size: usize, remove: usize) -> DropoutLayer {
        DropoutLayer {
            config: DropoutActivationConfig {
                size,
                remove
            },
            last_adjust: None
        }
    }

    pub const fn name() -> &'static str {
        "Dropout Activation"
    }
}

#[typetag::serde(name = "Dropout Layer")]
impl Layer for DropoutLayer {
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
        if training {
            let val = val.into_shape(self.config.size).unwrap();

            let adjust_by = (self.config.size as fXX) / ((self.config.size - self.config.remove) as fXX);
            let mut adjust = Array1::from_elem((self.config.size - self.config.remove,), adjust_by);
            adjust.append(Axis(0), Array1::zeros(self.config.remove).view()).unwrap();
            fastrand::shuffle(adjust.as_slice_mut().unwrap());

            let output = &val * &adjust;

            self.last_adjust = Some(adjust);

            output.into_dyn()
        }
        else {
            val
        }
    }

    fn backward_actual(&mut self, gradient: DATA, training_rate: fXX) -> DATA {
        gradient * self.last_adjust.take().unwrap()
    }

    fn data_bin(&self) -> Vec<Vec<u8>> {
        vec![]
    }

    fn load_data(&mut self, data: Vec<Vec<u8>>) {}
}
