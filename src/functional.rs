use crate::device::Device;
use crate::tensor::Tensor;
use crate::type_trait::{Float, Unsigned};

/// Computes the cross entropy loss between `logits` and `labels`.
/// **Warning**: `logits` should have the shape `[batch_size, num_classes]` and `labels` should have the shape `[batch_size]`.
pub fn cross_entropy_loss<T: Float, U: Unsigned, D: Device<T> + Device<U>>(
    logits: &Tensor<T, D>,
    labels: &Tensor<U, D>,
    mask: Option<Tensor<T, D>>,
) -> Tensor<T, D> {
    let mut shape = logits.shape();
    let num_classes = shape.pop().unwrap();
    let mut batch = T::from(shape.iter().product::<usize>()).unwrap();
    if let Some(mask) = mask.as_ref() {
        let t = mask.sum(None, false).underlying_data()[0];
        batch = batch / T::from(mask.realize_cached_data().len()).unwrap() * t;
    }
    let max_logits = logits.max(Some(vec![shape.len()]), true);
    let l = logits - &max_logits;
    let one_hot_labels = Tensor::one_hot(labels, num_classes, false);
    let p = (logits * &one_hot_labels).sum(Some(vec![shape.len()]), false);
    let mut loss =
        &(&l.exp().sum(Some(vec![shape.len()]), false).ln() + &max_logits.reshape(shape)) - &p;
    if let Some(mask) = mask {
        loss = &loss * &mask;
    }
    &loss.sum(None, false) / batch
}

pub fn top1_accuracy<T: Float, U: Unsigned, D: Device<T> + Device<U>>(
    logits: &Tensor<T, D>,
    labels: &Tensor<U, D>,
) -> T {
    let mut shape = logits.shape();
    let num_classes = shape.pop().unwrap();
    let batch: usize = shape.iter().product();
    let one_hot_labels = Tensor::one_hot(labels, num_classes, false);
    let p = (logits * &one_hot_labels).sum(Some(vec![shape.len()]), false);
    let q = logits.max(Some(vec![shape.len()]), false);
    (&p.equal(&q).sum(None, false) / T::from(batch).unwrap()).underlying_data()[0]
}

pub fn argmax<T: Float, U: Unsigned, D: Device<T> + Device<U>>(
    logits: &Tensor<T, D>,
) -> Tensor<U, D> {
    let data = logits.underlying_data();
    let mut shape = logits.shape();
    let num_classes = shape.pop().unwrap();
    let batch: usize = shape.iter().product();
    let mut argmax = Vec::<U>::with_capacity(batch);
    for i in 0..batch {
        let mut max = T::neg_infinity();
        let mut argmax_ = 0;
        for j in 0..num_classes {
            let x = data[i * num_classes + j];
            if x > max {
                max = x;
                argmax_ = j;
            }
        }
        argmax.push(U::from(argmax_).unwrap());
    }
    Tensor::new_with_shape(&argmax, &shape, false)
}

pub fn softmax<T: Float, D: Device<T>>(logits: &Tensor<T, D>, dim: usize) -> Tensor<T, D> {
    let max_logits = logits.max(Some(vec![dim]), true);
    let l = (logits - &max_logits).exp();
    &l / &l.sum(Some(vec![dim]), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CPU;

    #[test]
    fn test_cross_entropy_loss() {
        let logits = Tensor::<f32, CPU>::new2d([[0.0, 1.0], [2.0, 3.0]], false);
        let labels = Tensor::<u8, CPU>::new1d([0, 1], false);
        let loss = cross_entropy_loss(&logits, &labels, None);
        assert!(loss == Tensor::new1d([0.8132616875182228], false));
    }
}
