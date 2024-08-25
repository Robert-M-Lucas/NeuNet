use std::fs;
use std::fs::File;
use std::io::{stdout, Write};
use std::path::PathBuf;
use std::time::Instant;

use itertools::Itertools;
use ndarray::{Array, Axis, IxDyn};
use serde::{Deserialize, Serialize};

use crate::layers::Layer;
use crate::loss::{Loss, LossResult};

pub type fXX = f64;
pub type DATA = Array<fXX, IxDyn>;
pub type LAYERS = Vec<Box<dyn Layer>>;
pub type LOSS = Box<dyn Loss>;

#[derive(Clone)]
pub struct TrainingRateConfig {
    pub epochs: usize,
    pub initial_training_rate: fXX,
    pub final_training_rate: fXX
}

pub struct TrainingResult {
    pub loss: fXX,
    pub accuracy: fXX
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    layers: LAYERS,
    loss: LOSS
}

#[derive(Clone)]
pub struct LabeledData {
    pub data: DATA,
    pub labels: DATA
}

impl Model {
    pub fn new(model: LAYERS, loss: LOSS) -> Model {
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
            layers: model,
            loss
        }
    }

    pub fn forward(&mut self, input: DATA, training: bool) -> DATA {
        if self.layers.len() > 0 && input.shape() != &self.layers[0].input_shape() {
            panic!(
                "Data shape {:?} does not match first layer ({} [0]) input shape {:?}",
                input.shape(),
                self.layers[0].name(),
                self.layers[0].input_shape()
            )
        }

        if let Some((next, rest)) = self.layers.split_first_mut() {
            next.forward(input, rest, training)
        }
        else {
            input
        }
    }

    pub fn evaluate_kfold(
        model_generator: &dyn Fn() -> Model,
        full_data: LabeledData,
        per_fold_config: TrainingRateConfig,
        folds: usize,
        eval_fn: &dyn Fn(DATA, DATA) -> fXX,
    ) {
        let data_len = full_data.labels.len_of(Axis(0));
        let fold_size = data_len / folds;

        let mut data_folds =
            full_data.labels.axis_chunks_iter(Axis(0), fold_size)
                .zip(full_data.data.axis_chunks_iter(Axis(0), fold_size))
                .map(|(labels, data)|
                    LabeledData { labels: labels.into_owned(), data: data.into_owned() }
                )
                .collect_vec();

        // Merge last two folds if not perfectly divisible
        if data_folds.len() > folds {
            let LabeledData { data, labels } = data_folds.pop().unwrap();
            data_folds.last_mut().unwrap().labels.append(Axis(0), labels.view()).unwrap();
            data_folds.last_mut().unwrap().data.append(Axis(0), data.view()).unwrap();
        }

        println!("Starting k-fold cross-validation with folds: {:?}", data_folds.iter().map(|x| x.labels.len_of(Axis(0))).collect_vec());

        let mut total_accuracy = 0.0;
        for k in 0..folds {
            let mut model = model_generator();

            let mut testing_data = None;
            let mut training_data = None;

            for (nk, fold) in data_folds.iter().enumerate() {
                if k == nk {
                    testing_data = Some(fold.clone());
                }
                else {
                    if training_data.is_none() {
                        training_data = Some(fold.clone());
                    }
                    else {
                        training_data.as_mut().unwrap().labels.append(Axis(0), fold.labels.view()).unwrap();
                        training_data.as_mut().unwrap().data.append(Axis(0), fold.data.view()).unwrap();
                    }
                }
            }

            let testing_data = testing_data.unwrap();
            let training_data = training_data.unwrap();

            model.train(training_data, per_fold_config.clone(), Some(format!("[Fold {}/{folds}]", k + 1)));
            let accuracy = model.test(testing_data, eval_fn);
            total_accuracy += accuracy;
            println!("Fold {}/{folds} finished with an accuracy of {accuracy:.8}", k + 1);
        }

        println!("Model accuracy ({folds}-fold cross validation): {}", total_accuracy / folds as fXX);
    }

    pub fn train(
        &mut self,
        data: LabeledData,
        config: TrainingRateConfig,
        prefix: Option<String>
    ) {
        let a = (config.initial_training_rate - config.final_training_rate) / (1. - (1. / config.epochs as fXX));
        let c = config.initial_training_rate - a;
        let fx = |x: fXX| a * (1. / (x + 1.)) + c;

        let start = Instant::now();
        for e in 0..config.epochs {
            let training_rate = fx(e as fXX);
            let mut loss_total = 0.0;
            for (label, row) in data.labels.axis_iter(Axis(0)).zip(data.data.axis_iter(Axis(0))) {
                let prediction = self.forward(row.into_owned(), true);
                let clipped_prediction = prediction.mapv(|e| e.clamp(1e-7, 1.0 - 1e-7));
                let LossResult { loss, gradient } = self.loss.loss(clipped_prediction, label.into_owned());
                loss_total += loss;

                if let Some((next, rest)) = self.layers.as_mut_slice().split_last_mut() {
                    next.backward(gradient, training_rate, rest);
                }
            }
            let t = Instant::now() - start;
            let loss_avg = loss_total / data.labels.len_of(Axis(0)) as fXX;
            print!("\r");
            if let Some(prefix) = prefix.as_ref() { print!("{prefix} ") }
            print!("Epoch: {} | Loss: {loss_avg:.8} | Training Rate: {training_rate:.8} | Avg time per epoch: {:?}", e + 1, t / (e + 1) as u32);
            stdout().flush().unwrap();
        }
        println!();
    }

    pub fn test(
        &mut self,
        data: LabeledData,
        eval_fn: &dyn Fn(DATA, DATA) -> fXX,
    ) -> fXX {
        let mut total_accuracy = 0.0;

        for (label, row) in data.labels.axis_iter(Axis(0)).zip(data.data.axis_iter(Axis(0))) {
            let prediction = self.forward(row.into_owned(), false);
            total_accuracy += eval_fn(prediction, label.into_owned());
        }

        total_accuracy / data.labels.len_of(Axis(0)) as fXX
    }

    pub fn config(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
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

        let model: Model = serde_json::from_str(&config).unwrap();

        println!("Model '{}' loaded (no weights)", folder.display());

        model
    }

    pub fn load_with_weights(folder: &str) -> Model {
        let folder = PathBuf::from(folder);
        let weights = folder.join("weights");
        let config = fs::read_to_string(folder.join("config.json")).unwrap();

        let mut model: Model = serde_json::from_str(&config).unwrap();

        for (i, l) in model.layers.iter_mut().enumerate() {
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

        model
    }
}

struct TrainingBreakLayer {

}