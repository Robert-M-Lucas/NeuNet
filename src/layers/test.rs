// use serde::{Deserialize, Serialize};
// use typetag::serde;
//
// use crate::layers::{DATA, Layer};
// use crate::model::fXX;
//
// #[derive(Serialize, Deserialize)]
// pub struct TestLayerConfig {
//     val: fXX
// }
//
//
// #[derive(Serialize, Deserialize)]
// pub struct TestLayer {
//     #[serde(flatten)]
//     config: TestLayerConfig,
//     #[serde(skip)]
//     weights: Option<DATA>,
//     #[serde(skip)]
//     back_context: Option<()>
// }
//
// impl TestLayer {
//     pub fn new(val: fXX) -> TestLayer {
//         TestLayer {
//             config: TestLayerConfig {
//                 val,
//             },
//             weights: None,
//             back_context: None
//         }
//     }
//
//     pub const fn name() -> &'static str {
//         "TestLayer"
//     }
// }
//
// #[typetag::serde(name = "Test Layer")]
// impl Layer for TestLayer {
//     fn name(&self) -> &'static str {
//         Self::name()
//     }
//
//     fn input_shape(&self) -> Vec<usize> {
//         todo!()
//     }
//
//     fn output_shape(&self) -> Vec<usize> {
//         todo!()
//     }
//
//     fn forward_actual(&mut self, val: DATA, training: bool) -> DATA {
//         let mut out = val.mapv(|x| x + self.config.val);
//         if let Some(weights) = &self.weights {
//             out += weights;
//         }
//         self.weights = Some(out.clone());
//         out
//     }
//
// fn backward_actual(&mut self, gradient: DATA, training_rate: fXX) -> DATA {
//         todo!()
//     }
//
//     fn data_bin(&self) -> Vec<Vec<u8>> {
//         vec![bincode::serialize(self.weights.as_ref().unwrap()).unwrap()]
//     }
//
//     fn load_data(&mut self, data: Vec<Vec<u8>>) {
//         self.weights = Some(bincode::deserialize(&data[0]).unwrap());
//     }
// }
