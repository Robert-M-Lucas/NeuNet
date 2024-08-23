use std::path::Path;
use std::todo;
use serde::{Deserialize, Serialize};
use crate::layers::{DATA, Layer};

#[derive(Serialize, Deserialize)]
struct TestLayerConfig {
    val: i8
}

#[derive(Default)]
pub struct TestLayer {
    config: TestLayerConfig
}

impl TestLayer {
    pub const fn name() -> &'static str {
        "TestLayer"
    }

    pub fn from_config(file: &Path) -> Box<dyn Layer> {
        todo!()
    }
}

impl Layer for TestLayer {
    fn forward_actual(&mut self, val: DATA) -> DATA {
        val.mapv(|x| x + 1.0)
    }

    fn name(&self) -> &'static str {
        Self::name()
    }

    fn save(&self, file: &Path) {
        todo!()
    }

    fn save_weights(&self, file: &Path) {
        todo!()
    }

    fn load_model(file: &Path) {
        todo!()
    }

    fn load_weights(&self, file: &Path) {
        todo!()
    }

    fn inner(&self) -> &dyn Serialize {
        &self.config
    }
}
