use std::fs::File;
use b_box::b;
use csv::ReaderBuilder;
use itertools::Itertools;
use ndarray::{array, Array1, Array2, ArrayBase, Axis, OwnedRepr, s, stack, stack_new_axis};
use ndarray_csv::Array2Reader;

use crate::layers::dense::DenseLayer;
use crate::layers::dropout::DropoutLayer;
use crate::layers::relu::ReluActivator;
use crate::layers::softmax::SoftmaxActivator;
use crate::loss::mean_squared::MeanSquared;
use crate::model::{DATA, fXX, LabeledData, Model, TrainingRateConfig};

mod model;
mod layers;
mod loss;


// Generate model
fn fmodel_create() -> Model {
    let mut model = Model::new(vec![
        b!(DenseLayer::new_default(54, 54)),
        b!(ReluActivator::new(54)),
        b!(DropoutLayer::new(54, 0.25)),
        b!(DenseLayer::new_default(54, 54)),
        b!(ReluActivator::new(54)),
        b!(DropoutLayer::new(54, 0.2)),
        b!(DenseLayer::new_default(54, 27)),
        b!(ReluActivator::new(27)),
        b!(DropoutLayer::new(27, 0.1)),
        b!(DenseLayer::new_default(27, 2)),
        b!(SoftmaxActivator::new(2))
    ],
   b!(MeanSquared::new())
    );
    model
}

fn fmodel_load() -> Model {
    Model::load("models/model1")
}

fn fmodel_load_with_weights() -> Model {
    Model::load_with_weights("models/model1")
}

// Get data
fn fdata() -> LabeledData {
    let file = File::open("data/training_spam.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut array_read: ArrayBase<OwnedRepr<u8>, _> = reader.deserialize_array2((1000, 55)).unwrap().into_dyn();

    let file = File::open("data/testing_spam.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    array_read.append(Axis(0), reader.deserialize_array2((500, 55)).unwrap().into_dyn().view()).unwrap();

    let data = array_read.slice(s![.., ..54]).mapv(|e| e as fXX).into_dyn();

    // One-hot encode labels
    let hot: (ArrayBase<OwnedRepr<fXX>, _>, ArrayBase<OwnedRepr<fXX>, _>) = (array![1.0, 0.0], array![0.0, 1.0]);
    let labels = stack(Axis(0), &array_read.slice(s![.., 54]).iter().map(|e| if *e == 0 { hot.0.view() } else { hot.1.view() }).collect_vec()).unwrap().into_dyn();

    LabeledData {
        data,
        labels
    }
}

// Evaluate performance on hot encode. Give no 'marks' for being more ambiguous when wrong.
fn feval(output: DATA, actual: DATA) -> fXX {
    let output = output.into_shape(2).unwrap();
    let actual = actual.into_shape(2).unwrap();

    let fx = |x: Array1<fXX>| if x[0] > x[1] { false } else { true };

    if fx(output) == fx(actual) {
        1.0
    }
    else {
        0.0
    }
}

fn eval_model() {
    // ? Test model performance with k-fold
    let data = fdata();

    Model::evaluate_kfold(
        &fmodel_create,
        data,
        TrainingRateConfig {
            epochs: 5000,
            initial_training_rate: 0.1,
            final_training_rate: 0.0001,
        },
        6,
        &feval
    );
}

fn train() -> Model {
    // ? Train final model on all data
    let mut model = fmodel_create();

    let data = fdata();
    model.train(data, TrainingRateConfig {
        epochs: 5000,
        initial_training_rate: 0.1,
        final_training_rate: 0.0001,
    }, None);
    model
}

fn main() {
    eval_model();

    println!("\nTraining final model");
    let mut model = train();

    // Save trained model
    model.save_with_weights("models/main", true);
    // Save untrained model
    // model.save("models/model1", true);


    // Make prediction
    // let input = fdata().data.axis_iter(Axis(0)).nth(100).unwrap().into_owned();
    // let actual = fdata().labels.axis_iter(Axis(0)).nth(100).unwrap().into_owned();
    // let prediction = model.forward(input, false);

    // Terrible test on existing data
    println!("Accuracy: {}", model.test(fdata(), &feval));
}
