use crate::device::Device;
use crate::tensor::Tensor;
use crate::type_trait::{Float, Unsigned};

/// Computes the cross entropy loss between `logits` and `labels`.
/// **Warning**: `logits` should have the shape `[batch_size, num_classes]` and `labels` should have the shape `[batch_size]`.
pub fn cross_entropy_loss<T: Float, U: Unsigned, D: Device<T>, E: Device<U>>(
    logits: &Tensor<T, D>,
    labels: &Tensor<U, E>,
) -> Tensor<T, D> {
    let mut shape = logits.shape();
    let num_classes = shape.pop().unwrap();
    let batch: usize = shape.iter().product();
    let max_logits = logits.max(Some(vec![shape.len()]), true);
    let l = logits - &max_logits;
    let one_hot_labels = Tensor::one_hot(labels, num_classes, false);
    let p = (logits * &one_hot_labels).sum(Some(vec![shape.len()]), false);
    let loss =
        &(&l.exp().sum(Some(vec![shape.len()]), false).ln() + &max_logits.reshape(shape)) - &p;
    &loss.sum(None, false) / T::from(batch).unwrap()
}

pub fn top1_accuracy<T: Float, U: Unsigned, D: Device<T>, E: Device<U>>(
    logits: &Tensor<T, D>,
    labels: &Tensor<U, E>,
) -> T {
    let mut shape = logits.shape();
    let num_classes = shape.pop().unwrap();
    let batch: usize = shape.iter().product();
    let one_hot_labels = Tensor::one_hot(labels, num_classes, false);
    let p = (logits * &one_hot_labels).sum(Some(vec![shape.len()]), false);
    let q = logits.max(Some(vec![shape.len()]), false);
    (&p.equal(&q).sum(None, false) / T::from(batch).unwrap()).underlying_data()[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CPU;

    #[test]
    fn test_cross_entropy_loss() {
        let logits = Tensor::<f32, CPU>::new2d([[0.0, 1.0], [2.0, 3.0]], false);
        let labels = Tensor::<u8, CPU>::new1d([0, 1], false);
        let loss = cross_entropy_loss(&logits, &labels);
        assert!(loss == Tensor::new1d([0.8132616875182228], false));
    }
}
