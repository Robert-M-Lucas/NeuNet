use serde::{Deserialize, Serialize};
use typetag::serde;

use crate::layers::{DATA, Layer};

#[derive(Serialize, Deserialize)]
pub struct TestLayerConfig {
    val: f64
}


#[derive(Serialize, Deserialize)]
pub struct TestLayer {
    #[serde(flatten)]
    config: TestLayerConfig,
    #[serde(skip)]
    weights: Option<DATA>
}

impl TestLayer {
    pub fn new(val: f64) -> TestLayer {
        TestLayer {
            config: TestLayerConfig {
                val,
            },
            weights: None
        }
    }

    pub const fn name() -> &'static str {
        "TestLayer"
    }
}

#[typetag::serde]
impl Layer for TestLayer {
    fn forward_actual(&mut self, val: DATA) -> DATA {
        let mut out = val.mapv(|x| x + self.config.val);
        if let Some(weights) = &self.weights {
            out += weights;
        }
        self.weights = Some(out.clone());
        out
    }

    fn name(&self) -> &'static str {
        Self::name()
    }

    fn weights_bin(&self) -> Vec<u8> {
        bincode::serialize(self.weights.as_ref().unwrap()).unwrap()
    }

    fn load_weights(&mut self, weights_bin: Vec<u8>) {
        self.weights = Some(bincode::deserialize(&weights_bin).unwrap());
    }

}
