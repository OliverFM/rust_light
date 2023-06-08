use crate::tensor::RcTensor;

// These are all aliases that should probably be converted to traits/structs at somepoint
// However they exist now for the sake of readability.
// I have vague hopes of going in an improving this in future.
// The main reason to avoid doing this now is that I think the API and library
// is still far too unstable for me to commit to anything yet
pub type TensorList<T> = Vec<RcTensor<T>>;

pub type Scalar<T> = RcTensor<T>;
