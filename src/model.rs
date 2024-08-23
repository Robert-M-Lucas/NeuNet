use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use itertools::Itertools;
use ndarray::{Array, IxDyn};

use crate::layers::Layer;

pub type DataElement = f64;
pub type DATA = Array<DataElement, IxDyn>;
pub type LAYERS = Vec<Box<dyn Layer>>;

pub struct Model {
    layers: LAYERS,
}

impl Model {
    pub fn new(model: LAYERS) -> Model {
        for ((i, layer), next_layer) in model.iter().enumerate().zip(model.iter().skip(1)) {
            if layer.output_shape() != next_layer.input_shape() {
                panic!(
                    "{} [{}] with output shape {:?} does not match {} [{}] with input shape {:?}",
                    layer.name(),
                    i,
                    layer.output_shape(),
                    next_layer.name(),
                    i + 1,
                    next_layer.input_shape()
                )
            }
        }

        Model {
            layers: model
        }
    }

    pub fn forward(&mut self, input: DATA) -> DATA {
        if self.layers.len() > 0 && input.shape() != &self.layers[0].input_shape() {
            panic!(
                "Data shape {:?} does not match first layer ({} [0]) input shape {:?}",
                input.shape(),
                self.layers[0].name(),
                self.layers[0].input_shape()
            )
        }

        if let Some((next, rest)) = self.layers.split_first_mut() {
            next.forward(input, rest, false)
        }
        else {
            input
        }
    }

    pub fn forward_with_context(&mut self, input: DATA) -> DATA {
        if let Some((next, rest)) = self.layers.split_first_mut() {
            next.forward(input, rest, true)
        }
        else {
            input
        }
    }

    pub fn config(&self) -> String {
        serde_json::to_string_pretty(&self.layers).unwrap()
    }

    pub fn save(&self, folder: &str, overwrite: bool) {
        let folder = PathBuf::from(folder);
        if folder.is_dir() {
            if overwrite {
                fs::remove_dir_all(&folder).unwrap();
            }
            else {
                panic!("Model '{}' already exists!", folder.display())
            }
        }

        fs::create_dir_all(&folder).unwrap();

        let config_file = folder.join("config.json");
        fs::write(config_file, self.config()).unwrap();

        println!("Model '{}' saved (no weights)", folder.display());
    }

    pub fn save_with_weights(&self, folder: &str, overwrite: bool) {
        let folder = PathBuf::from(folder);
        let weights = folder.join("weights");
        if folder.is_dir() {
            if overwrite {
                fs::remove_dir_all(&folder).unwrap();
            }
            else {
                panic!("Model '{}' already exists!", folder.display())
            }
        }

        fs::create_dir_all(&folder).unwrap();

        let config_file = folder.join("config.json");
        fs::write(config_file, self.config()).unwrap();

        fs::create_dir(&weights).unwrap();
        for (i, l) in self.layers.iter().enumerate() {
            let data = l.data_bin();

            let subfolder = weights.join(format!("{i}"));
            fs::create_dir(&subfolder).unwrap();

            for (i, section) in data.iter().enumerate() {
                let file_name = format!("{i}.dat");
                let mut file = File::create(subfolder.join(file_name)).unwrap();
                file.write(section).unwrap();
            }
        }

        println!("Model '{}' saved (with weights)", folder.display());
    }

    pub fn load(folder: &str) -> Model {
        let folder = PathBuf::from(folder);
        let config = fs::read_to_string(folder.join("config.json")).unwrap();

        let layers: LAYERS = serde_json::from_str(&config).unwrap();

        println!("Model '{}' loaded (no weights)", folder.display());

        Model {
            layers
        }

    }

    pub fn load_with_weights(folder: &str) -> Model {
        let folder = PathBuf::from(folder);
        let weights = folder.join("weights");
        let config = fs::read_to_string(folder.join("config.json")).unwrap();

        let mut layers: LAYERS = serde_json::from_str(&config).unwrap();

        for (i, l) in layers.iter_mut().enumerate() {
            let subfolder = weights.join(format!("{i}"));

            let mut data = Vec::new();

            for i in 0usize.. {
                let file = subfolder.join(format!("{i}.dat"));
                if file.exists() {
                    data.push(fs::read(file).unwrap());
                }
                else { break; }
            }

            l.load_data(data);
        }

        println!("Model '{}' loaded (with weights)", folder.display());

        Model {
            layers
        }
    }
}