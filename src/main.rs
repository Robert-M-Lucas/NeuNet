use ndarray::array;

use crate::layers::{Layer, Model};

mod layers;



fn main() {
    // let mut model = Model::new(vec![
    //     b!(TestLayer::new(1.0)),
    //     b!(TestLayer::new(2.5)),
    // ]);
    //
    let mut model = Model::load_with_weights("models/model1");


    println!("{}", model.forward(array![1.0, 2.0, 3.0].into_dyn()));

    model.save_with_weights("models/model1", true);
}
