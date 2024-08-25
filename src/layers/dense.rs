use ndarray::{array, Array, Array1, Array2, Axis, s};
use ndarray_rand::rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};
use typetag::serde;
use ndarray_rand::RandomExt;

use crate::layers::{DATA, Layer};
use crate::model::fXX;

pub enum WeightMode {
    Equal,
    Normal,
    Custom(Array2<fXX>)
}

#[derive(Serialize, Deserialize)]
pub struct DenseLayerConfig {
    input_size: usize,
    output_size: usize
}

#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    #[serde(flatten)]
    config: DenseLayerConfig,
    #[serde(skip)]
    weights: Array2<fXX>,
    #[serde(skip)]
    biases: Array1<fXX>,
    #[serde(skip)]
    back_context: Option<Array1<fXX>>
}

impl DenseLayer {
    pub fn new_default(input_size: usize, output_size: usize) -> DenseLayer {
        Self::new(input_size, output_size, WeightMode::Normal, 0.0)
    }

    pub fn new(input_size: usize, output_size: usize, initial_weight: WeightMode, initial_bias: fXX) -> DenseLayer {
        let weights = match initial_weight {
            WeightMode::Equal => {
                Array2::from_elem((input_size, output_size), 1.0 / input_size as fXX)
            }
            WeightMode::Normal => {
                // Variance adds so variance should sum to constant. Divide all variance by count and sqrt to get std dev
                const VARIANCE_TARGET: fXX = 0.5;
                Array::random((input_size, output_size), Normal::new(0., (VARIANCE_TARGET / input_size as fXX).sqrt()).unwrap())
            }
            WeightMode::Custom(w) => w,
        };

        DenseLayer {
            config: DenseLayerConfig {
                input_size,
                output_size
            },
            weights,
            biases: Array1::from_elem((output_size,), initial_bias),
            back_context: None,
        }
    }

    pub const fn name() -> &'static str {
        "Dense Layer"
    }
}

#[typetag::serde(name = "Dense Layer")]
impl Layer for DenseLayer {
    fn name(&self) -> &'static str {
        Self::name()
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![self.config.input_size]
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.config.output_size]
    }

    fn forward_actual(&mut self, val: DATA, training: bool) -> DATA {
        let val = val.into_shape((self.config.input_size,)).unwrap();
        let output = val.dot(&self.weights) + &self.biases;

        if training {
            self.back_context = Some(val.into_owned());
        }

        output.into_dyn()
    }

    fn backward_actual(&mut self, gradient: DATA, training_rate: fXX) -> DATA {
        // TODO: Max gradient?

        let gradient = gradient.into_shape(self.config.output_size).unwrap();
        self.biases = &self.biases - (&gradient * training_rate);

        let back_context = self.back_context.take().unwrap();
        for (mut wa, bc) in self.weights.axis_iter_mut(Axis(0)).zip(back_context.iter()) {
            wa -= &((&gradient * *bc) * training_rate).view();
        }

        // self.weights.slice -= kron(self.back_context.unwrap(), &gradient) * training_rate;

        Array::from_iter(self.weights.axis_iter(Axis(0)).map(|r| r.dot(&gradient))).into_dyn()
    }

    fn data_bin(&self) -> Vec<Vec<u8>> {
        vec![
            bincode::serialize(&self.weights).unwrap(),
            bincode::serialize(&self.biases).unwrap(),
        ]
    }

    fn load_data(&mut self, data: Vec<Vec<u8>>) {
        self.weights = bincode::deserialize(&data[0]).unwrap();
        self.biases = bincode::deserialize(&data[1]).unwrap();
    }
}
