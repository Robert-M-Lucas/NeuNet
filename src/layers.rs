use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use itertools::Itertools;
use ndarray::{Array, IxDyn};
use serde::{Deserialize, Serialize};

pub mod test_layer;

pub type DATA = Array<f64, IxDyn>;
pub type LAYERS = Vec<Box<dyn Layer>>;

pub struct Model {
    layers: LAYERS
}

impl Model {
    pub fn new(model: LAYERS) -> Model {
        Model {
            layers: model
        }
    }

    pub fn forward(&mut self, input: DATA) -> DATA {
        if let Some((next, rest)) = self.layers.split_first_mut() {
            next.forward(input, rest)
        }
        else {
            input
        }
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
        fs::write(config_file, serde_json::to_string_pretty(&self.layers).unwrap()).unwrap();

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
        fs::write(config_file, serde_json::to_string_pretty(&self.layers).unwrap()).unwrap();

        fs::create_dir(&weights).unwrap();
        for (i, l) in self.layers.iter().enumerate() {
            let file_name = format!("{i}.dat");
            let mut file = File::create(weights.join(file_name)).unwrap();
            file.write(&l.weights_bin()).unwrap();
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

        for i in 0..layers.len() {
            let file_name = format!("{i}.dat");
            let file = fs::read(weights.join(file_name)).unwrap();
            layers[i].load_weights(file);
        }

        println!("Model '{}' loaded (with weights)", folder.display());

        Model {
            layers
        }
    }
}


#[typetag::serde]
pub trait Layer {
    fn forward(&mut self, val: DATA, layers: &mut [Box<dyn Layer>]) -> DATA {
        let new_val = self.forward_actual(val);

        if let Some((next, rest)) = layers.split_first_mut() {
            next.forward(new_val, rest)
        }
        else {
            new_val
        }
    }

    fn forward_actual(&mut self, val: DATA) -> DATA;

    fn name(&self) -> &'static str;

    fn weights_bin(&self) -> Vec<u8>;

    fn load_weights(&mut self, weights_bin: Vec<u8>);
}