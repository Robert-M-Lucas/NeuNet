use crate::model::DATA;

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

    fn forward(&mut self, val: DATA, layers: &mut [Box<dyn Layer>], save_context: bool) -> DATA {
        let new_val = self.forward_actual(val, save_context);

        if let Some((next, rest)) = layers.split_first_mut() {
            next.forward(new_val, rest, save_context)
        }
        else {
            new_val
        }
    }

    fn forward_actual(&mut self, val: DATA, save_context: bool) -> DATA;

    fn data_bin(&self) -> Vec<Vec<u8>>;

    fn load_data(&mut self, data: Vec<Vec<u8>>);
}