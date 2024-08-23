use b_box::b;
use ndarray::array;
use crate::layers::dense::DenseLayer;
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
        b!(DenseLayer::new_default(54, 27)),
        b!(ReluActivator::new(27)),
        b!(DenseLayer::new_default(27, 2)),
        b!(SoftmaxActivator::new(2))
    ]);

    // let mut model = Model::load_with_weights("models/model1");

    println!("Model:\n{}", model.config());

    let result = model.forward(
        array![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
            .into_dyn()
    );

    println!("{}", result);

    model.save_with_weights("models/model1", true);
}
