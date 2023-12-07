use crate::device::Device;
use crate::ndarray::NDArray;
use crate::tensor::Tensor;
use crate::type_trait::Type;

pub(crate) trait Operation<'a, T: Type + 'a, D: Device<T>> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D>;
    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>>;
}

pub(crate) struct Broadcast(pub Vec<usize>);

pub(crate) struct Summation(pub Option<Vec<usize>>, pub bool);

pub(crate) struct Reshape(pub Vec<usize>);

pub(crate) struct EWiseAdd;

pub(crate) struct AddScalar<T: Type>(pub T);

pub(crate) struct EWiseMul;

pub(crate) struct MulScalar<T: Type>(pub T);

pub(crate) struct EWisePow;

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for Broadcast {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].broadcast(&self.0)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        vec![reduce_by_add(out_grad, &node.shape())]
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for Summation {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].sum(self.0.clone(), self.1)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        if self.1 || self.0.is_none() {
            vec![out_grad.broadcast(&node.shape())]
        } else {
            let mut shape = node.shape();
            for &axis in self.0.as_ref().unwrap() {
                shape[axis] = 1;
            }
            vec![out_grad.reshape(shape).broadcast(&node.shape())]
        }
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for Reshape {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].reshape(&self.0)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        vec![out_grad.reshape(node.shape())]
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for EWiseAdd {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs + rhs)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let grad_shape = out_grad.shape();
        let mut in_grads = vec![];
        for i in &node.0.write().unwrap().inputs {
            let shape = i.shape();
            in_grads.push({
                if shape == grad_shape {
                    out_grad.clone()
                } else {
                    reduce_by_add(&out_grad, &shape)
                }
            });
        }
        in_grads
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for AddScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        &args[0] + self.0
    }

    fn gradient(&self, out_grad: &Tensor<'a, T, D>, _: &Tensor<'a, T, D>) -> Vec<Tensor<'a, T, D>> {
        vec![out_grad.clone()]
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for EWiseMul {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs * rhs)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let grad_shape = out_grad.shape();
        let mut in_grads = vec![];
        let inputs = &node.0.write().unwrap().inputs;
        for (i, input) in inputs.iter().enumerate() {
            let shape = input.shape();
            in_grads.push({
                let in_grad = out_grad * &inputs[1 - i];
                if shape == grad_shape {
                    in_grad
                } else {
                    reduce_by_add(&in_grad, &shape)
                }
            });
        }
        in_grads
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for MulScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        &args[0] * self.0
    }

    fn gradient(&self, out_grad: &Tensor<'a, T, D>, _: &Tensor<'a, T, D>) -> Vec<Tensor<'a, T, D>> {
        vec![out_grad * self.0]
    }
}

impl<'a, T: Type + 'a, D: Device<T>> Operation<'a, T, D> for EWisePow {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs * rhs)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let grad_shape = out_grad.shape();
        let mut in_grads = vec![];
        let inputs = &node.0.write().unwrap().inputs;
        for (i, input) in inputs.iter().enumerate() {
            let shape = input.shape();
            in_grads.push({
                let in_grad = out_grad * &inputs[1 - i];
                if shape == grad_shape {
                    in_grad
                } else {
                    reduce_by_add(&in_grad, &shape)
                }
            });
        }
        in_grads
    }
}

fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    if lhs.len() > rhs.len() {
        broadcast_shapes(rhs, lhs)
    } else {
        let mut shape = rhs.to_vec();
        let p = rhs.len() - lhs.len();
        for (i, &dim) in lhs.iter().enumerate() {
            if shape[i + p] != dim {
                if shape[i + p] == 1 {
                    shape[i + p] = dim;
                } else if dim != 1 {
                    panic!(
                        "Operands could not be broadcast together with shapes {:?} {:?}",
                        lhs, rhs
                    );
                }
            }
        }
        shape
    }
}

fn reduce_by_add<'a, T: Type, D: Device<T>>(
    input: &Tensor<'a, T, D>,
    output_shape: &[usize],
) -> Tensor<'a, T, D> {
    let input_shape = input.shape();
    let n = input_shape.len() - output_shape.len();
    let sum = input.sum(Some((0..n).collect::<Vec<_>>()), false);
    let mut reduced_axes = vec![];
    for i in n..input_shape.len() {
        if input_shape[i] != output_shape[i - n] {
            reduced_axes.push(i - n);
        }
    }
    if reduced_axes.is_empty() {
        sum
    } else {
        sum.sum(Some(reduced_axes), true)
    }
}

fn apply_with_broadcast<T: Type, D: Device<T>>(
    args: &[NDArray<T, D>],
    op: impl Fn(&NDArray<T, D>, &NDArray<T, D>) -> NDArray<T, D>,
) -> NDArray<T, D> {
    let mut lhs = &args[0];
    let mut rhs = &args[1];
    let lhs_reshape;
    let rhs_reshape;
    if args[0].0.shape != args[1].0.shape {
        let shape = broadcast_shapes(&args[0].0.shape, &args[1].0.shape);
        if shape != args[0].0.shape {
            lhs_reshape = args[0].broadcast(&shape);
            lhs = &lhs_reshape;
        }
        if shape != args[1].0.shape {
            rhs_reshape = args[1].broadcast(&shape);
            rhs = &rhs_reshape;
        }
    }
    op(lhs, rhs)
}
