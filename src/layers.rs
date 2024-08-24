use crate::model::{DATA, fXX};

pub mod test;
pub mod relu;
pub mod dense;
pub mod softmax;
pub mod dropout;

#[typetag::serde]
pub trait Layer {
    fn name(&self) -> &'static str;

    fn input_shape(&self) -> Vec<usize>;

    fn output_shape(&self) -> Vec<usize>;

    fn forward(&mut self, val: DATA, layers: &mut [Box<dyn Layer>], training: bool) -> DATA {
        let new_val = self.forward_actual(val, training);

        if let Some((next, rest)) = layers.split_first_mut() {
            next.forward(new_val, rest, training)
        }
        else {
            new_val
        }
    }

    fn forward_actual(&mut self, val: DATA, training: bool) -> DATA;

    fn backward(&mut self, gradient: DATA, training_rate: fXX, layers: &mut [Box<dyn Layer>]) -> DATA {
        let new_gradient = self.backward_actual(gradient, training_rate);

        if let Some((next, rest)) = layers.split_last_mut() {
            next.backward(new_gradient, training_rate, rest)
        }
        else {
            new_gradient
        }
    }

    fn backward_actual(&mut self, gradient: DATA, training_rate: fXX) -> DATA;



    fn data_bin(&self) -> Vec<Vec<u8>>;

    fn load_data(&mut self, data: Vec<Vec<u8>>);
}