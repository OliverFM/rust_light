use crate::tensor::{Numeric, TensorList};

pub(crate) mod private {
    pub trait Private {}
}

pub trait Module<T: Numeric>: private::Private {
    fn forward(&self, inputs: TensorList<T>) -> TensorList<T>;
}
