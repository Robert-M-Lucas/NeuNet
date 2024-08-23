use std::hint::black_box;
use std::time::Instant;
use b_box::b;
use ndarray::{array, Array1, ArrayBase, Axis, OwnedRepr, s};
use crate::layers::dense::DenseLayer;
use crate::layers::dropout::DropoutLayer;
use crate::layers::relu::ReluActivator;
use crate::layers::softmax::SoftmaxActivator;
use crate::layers::test::TestLayer;
use crate::model::Model;

mod model;
mod layers;


fn main() {
    let mut model = Model::new(vec![
        b!(DenseLayer::new_default(54, 54)),
        b!(ReluActivator::new(54)),
        // b!(DropoutLayer::new(54, 0.2)),
        b!(DenseLayer::new_default(54, 27)),
        b!(ReluActivator::new(27)),
        // b!(DropoutLayer::new(27, 0.1)),
        b!(DenseLayer::new_default(27, 2)),
        b!(SoftmaxActivator::new(2))
    ]);

    // let mut model = Model::load_with_weights("models/model1");

    println!("Model:\n{}", model.config());

    let result = model.forward_with_context(
        array![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
            .into_dyn()
    );

    println!("{}", result);

    model.save_with_weights("models/model1", true);
}
