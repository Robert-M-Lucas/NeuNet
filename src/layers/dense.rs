use ndarray::{array, Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use typetag::serde;

use crate::layers::{DATA, Layer};
use crate::model::DataElement;

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
    weights: Array2<DataElement>,
    #[serde(skip)]
    biases: Array1<DataElement>,
    #[serde(skip)]
    back_context: Option<Array1<DataElement>>
}

impl DenseLayer {
    pub fn new_default(input_size: usize, output_size: usize) -> DenseLayer {
        Self::new(input_size, output_size, 1.0 / input_size as DataElement, 0.0)
    }

    pub fn new(input_size: usize, output_size: usize, initial_weight: DataElement, initial_bias: DataElement) -> DenseLayer {
        DenseLayer {
            config: DenseLayerConfig {
                input_size,
                output_size
            },
            weights: Array2::from_elem((input_size, output_size), initial_weight),
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

    fn forward_actual(&mut self, val: DATA, save_context: bool) -> DATA {
        let val = val.to_shape((self.config.input_size,)).unwrap();
        let output = val.dot(&self.weights) + &self.biases;

        if save_context {
            self.back_context = Some(val.into_owned());
        }

        output.into_dyn()
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
