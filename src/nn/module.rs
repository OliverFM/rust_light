use crate::tensor::{Numeric, TensorList};

pub(crate) mod private {
    pub trait Private {}
}

pub trait Module<T: Numeric>: private::Private {
    type InputType;
    type OutputType;
    // TODO: figure out how to encode that these must be some kind of TensorLike or TensorList
    // I could use enums, but I don't want the user to need to wrap the inputs
    fn forward(&self, inputs: Self::InputType) -> Self::OutputType;

    fn params(&self) -> TensorList<T>;
}
