use crate::device::Device;
use crate::ndarray::NDArray;
use crate::tensor::Tensor;
use crate::type_trait::{Float, Signed, Type, Unsigned};
use num_traits::{One, Pow};
use std::ops::Div;

pub(crate) trait Operation<T: Type, D: Device<T>> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D>;

    /// Reversed automatic differentiation implementation.
    /// **Warning**: This function can't get the `RwLockWriteGuard` of `node`!
    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>>;
}

pub(crate) struct Broadcast(pub Vec<usize>);

pub(crate) struct Summation(pub Option<Vec<usize>>, pub bool);

pub(crate) struct Reshape(pub Vec<usize>);

pub(crate) struct EWiseAdd;

pub(crate) struct AddScalar<T: Type>(pub T);

pub(crate) struct EWiseMul;

pub(crate) struct MulScalar<T: Type>(pub T);

pub(crate) struct EWiseSub;

pub(crate) struct ScalarSub<T: Type>(pub T);

pub(crate) struct EWisePow;

pub(crate) struct PowScalar<T: Type>(pub T);

pub(crate) struct ScalarPow<T: Type>(pub T);

pub(crate) struct EWiseDiv;

pub(crate) struct DivScalar<T: Type>(pub T);

pub(crate) struct ScalarDiv<T: Type>(pub T);

pub(crate) struct Ln;

pub(crate) struct Sqrt;

pub(crate) struct Transpose(pub Option<(usize, usize)>);

pub(crate) struct MaximumScalar<T: Type>(pub T);

pub(crate) struct GTScalar<T: Type>(pub T);

pub(crate) struct Matmul;

pub(crate) struct Max(pub Option<Vec<usize>>, pub bool);

pub(crate) struct Equal;

pub(crate) struct Index<U: Unsigned, F: Device<U>>(pub Tensor<U, F>);

pub(crate) struct IndexRev<U: Unsigned, F: Device<U>>(pub Tensor<U, F>, pub usize);

pub(crate) struct Cat(pub usize);

pub(crate) struct Split(pub usize, pub usize, pub usize);

pub(crate) struct Sin;

pub(crate) struct Cos;

impl<T: Type, D: Device<T>> Operation<T, D> for Broadcast {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].broadcast(&self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![reduce_by_add(out_grad, node.data().unwrap().shape())]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Summation {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].sum(self.0.clone(), self.1)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        if self.1 || self.0.is_none() {
            vec![out_grad.broadcast(&node.0.read().unwrap().inputs[0].shape())]
        } else {
            let ori_shape = node.0.read().unwrap().inputs[0].shape();
            let mut shape = ori_shape.clone();
            for &axis in self.0.as_ref().unwrap() {
                shape[axis] = 1;
            }
            vec![out_grad.reshape(shape).broadcast(&ori_shape)]
        }
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Max {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].max(self.0.clone(), self.1)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        if self.1 || self.0.is_none() {
            vec![out_grad * &(node.equal(&node.0.read().unwrap().inputs[0]))]
        } else {
            let mut shape = node.0.read().unwrap().inputs[0].shape();
            for &axis in self.0.as_ref().unwrap() {
                shape[axis] = 1;
            }
            vec![
                &out_grad.reshape(shape.clone())
                    * &(node.reshape(shape).equal(&node.0.read().unwrap().inputs[0])),
            ]
        }
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Reshape {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].reshape(&self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad.reshape(node.0.read().unwrap().inputs[0].shape())]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Equal {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs.equal(rhs))
    }

    fn gradient(&self, _: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        vec![
            Tensor::zeros_like(&inputs[0], false),
            Tensor::zeros_like(&inputs[1], false),
        ]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for EWiseAdd {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs + rhs)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        let in_grads = vec![out_grad.clone(), out_grad.clone()];
        reduce_to_shape(in_grads, inputs)
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for AddScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        &args[0] + self.0
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, _: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad.clone()]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for EWiseSub {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs - rhs)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        let in_grads = vec![out_grad.clone(), -out_grad];
        reduce_to_shape(in_grads, inputs)
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for ScalarSub<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].scalar_sub(self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, _: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![-out_grad]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for EWiseMul {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs * rhs)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        let in_grads = vec![out_grad * &inputs[1], out_grad * &inputs[0]];
        reduce_to_shape(in_grads, inputs)
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for MulScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        &args[0] * self.0
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, _: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad * self.0]
    }
}

impl<T: Float, D: Device<T>> Operation<T, D> for EWisePow {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs.pow(rhs))
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        let in_grads = vec![
            &(out_grad * &inputs[1]) * &inputs[0].pow(&(&inputs[1] - T::one())),
            &(out_grad * &inputs[0].pow(&inputs[1])) * &inputs[1].ln(),
        ];
        reduce_to_shape(in_grads, inputs)
    }
}

impl<D: Device<T>, T: Float> Operation<T, D> for ScalarPow<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].scalar_pow(self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let input = &node.0.read().unwrap().inputs[0];
        vec![&(out_grad * &self.0.powt(input)) * self.0.ln()]
    }
}

impl<T: Float, D: Device<T>> Operation<T, D> for PowScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].pow(self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let input = &node.0.read().unwrap().inputs[0];
        vec![&(out_grad * self.0) * &input.pow(self.0 - T::one())]
    }
}

macro_rules! impl_pow_scalar {
    ($t:ty, $u:ty) => {
        impl<D: Device<$t>> Operation<$t, D> for PowScalar<$u> {
            fn compute(&self, args: &[NDArray<$t, D>]) -> NDArray<$t, D> {
                args[0].pow(self.0)
            }

            fn gradient(
                &self,
                out_grad: &Tensor<$t, D>,
                node: &Tensor<$t, D>,
            ) -> Vec<Tensor<$t, D>> {
                let input = &node.0.read().unwrap().inputs[0];
                vec![&(out_grad * (self.0 as $t)) * &input.pow(self.0 - <$u as One>::one())]
            }
        }
    };
}

impl_pow_scalar!(f32, i32);
impl_pow_scalar!(f64, i32);
impl_pow_scalar!(isize, u32);
impl_pow_scalar!(i8, u32);
impl_pow_scalar!(i16, u32);
impl_pow_scalar!(i32, u32);
impl_pow_scalar!(i64, u32);

impl<T: Signed, D: Device<T>> Operation<T, D> for EWiseDiv {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs.div(rhs))
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        let in_grads = vec![
            out_grad / &inputs[1],
            &(out_grad * &-&inputs[0]) / &(&inputs[1] * &inputs[1]),
        ];
        reduce_to_shape(in_grads, inputs)
    }
}

impl<T: Signed, D: Device<T>> Operation<T, D> for DivScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].div(self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, _: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad / self.0]
    }
}

impl<D: Device<T>, T: Signed> Operation<T, D> for ScalarDiv<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].scalar_div(self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let input = &node.0.read().unwrap().inputs[0];
        vec![&(out_grad * -self.0) / &(input * input)]
    }
}

impl<T: Float, D: Device<T>> Operation<T, D> for Ln {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].ln()
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad / &node.0.read().unwrap().inputs[0]]
    }
}

impl<T: Float, D: Device<T>> Operation<T, D> for Sqrt {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].sqrt()
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad / &(node * T::from(2).unwrap())]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Transpose {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        let len = args[0].ndim();
        if len < 2 {
            panic!("Transpose requires at least 2 dimensions");
        }
        let mut axes = (0..len).collect::<Vec<_>>();
        if let Some((i, j)) = self.0 {
            axes.swap(i, j);
        } else {
            axes.swap(len - 1, len - 2);
        };
        args[0].permute(&axes)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, _: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad.transpose(self.0)]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for MaximumScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].max_scalar(self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad * &node.gt(self.0)]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for GTScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].gt_scalar(self.0)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, _: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad * T::zero()]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Matmul {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        let mut lhs = &args[0];
        let mut rhs = &args[1];
        let lhs_shape = &lhs.0.shape;
        let rhs_shape = &rhs.0.shape;
        let lhs_len = lhs_shape.len();
        let rhs_len = rhs_shape.len();
        if lhs_len < 2 || rhs_len < 2 {
            panic!("Matmul requires at least 2 dimensions");
        }
        if lhs_shape[lhs_len - 1] != rhs_shape[rhs_len - 2] {
            panic!("Incompatible shapes: {:?} {:?}", lhs_shape, rhs_shape);
        }
        let lhs_reshape;
        let rhs_reshape;
        let lhs_shared_shape = &lhs_shape[..lhs_len - 2];
        let rhs_shared_shape = &rhs_shape[..rhs_len - 2];
        let [n, m, k] = [
            lhs_shape[lhs_len - 2],
            lhs_shape[lhs_len - 1],
            rhs_shape[rhs_len - 1],
        ];
        if lhs_shared_shape != rhs_shared_shape {
            let shape = broadcast_shapes(&lhs_shared_shape, &rhs_shared_shape);
            if shape != lhs_shared_shape {
                lhs_reshape = args[0].broadcast(&[shape.clone(), vec![n, m]].concat());
                lhs = &lhs_reshape;
            }
            if shape != rhs_shared_shape {
                rhs_reshape = args[1].broadcast(&[shape, vec![m, k]].concat());
                rhs = &rhs_reshape;
            }
        }
        lhs.matmul(rhs)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        let in_grads = vec![
            out_grad.matmul(&inputs[1].transpose(None)),
            inputs[0].transpose(None).matmul(out_grad),
        ];
        reduce_to_shape(in_grads, inputs)
    }
}

impl<T: Type, U: Unsigned, D: Device<T> + Device<U> + 'static> Operation<T, D> for Index<U, D> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].index(self.0.realize_cached_data())
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad.index_rev(&self.0, node.0.read().unwrap().inputs[0].shape()[0])]
    }
}

impl<T: Type, U: Unsigned, D: Device<T> + Device<U> + 'static> Operation<T, D> for IndexRev<U, D> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].index_rev(self.0.realize_cached_data(), self.1)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, _: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad.index(&self.0)]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Cat {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        let mut shape = args[0].shape().to_vec();
        if self.0 >= shape.len() {
            panic!("Invalid dimension");
        }
        for i in &args[1..] {
            let mut arg_shape = i.shape().to_vec();
            if self.0 >= arg_shape.len() {
                panic!("Invalid dimension");
            }
            shape[self.0] += arg_shape[self.0];
            arg_shape[self.0] = shape[self.0];
            if arg_shape != shape {
                panic!("Invalid shape");
            }
        }
        NDArray::cat(args, self.0, shape)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let inputs = &node.0.read().unwrap().inputs;
        let lhs_dim = inputs[0].shape()[self.0];
        let rhs_dim = inputs[1].shape()[self.0];
        vec![
            out_grad.split(self.0, 0, lhs_dim),
            out_grad.split(self.0, lhs_dim, rhs_dim),
        ]
    }
}

impl<T: Type, D: Device<T>> Operation<T, D> for Split {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        if args[0].ndim() <= self.0 {
            panic!("Invalid dimension");
        }
        let dim = args[0].shape()[self.0];
        if dim < self.1 + self.2 {
            panic!("Invalid shape");
        }
        args[0].split(self.0, self.1, self.2)
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        let input = &node.0.read().unwrap().inputs[0];
        let mut shape = input.shape();
        shape[self.0] -= self.1 + self.2;
        let requires_grad = out_grad.0.read().unwrap().requires_grad;
        let third = Tensor::zeros(&shape, requires_grad);
        shape[self.0] = self.1;
        let first = Tensor::zeros(&shape, requires_grad);
        vec![first.cat(&[out_grad, &third], self.0)]
    }
}

impl<T: Float, D: Device<T>> Operation<T, D> for Sin {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].sin()
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad * &node.0.read().unwrap().inputs[0].cos()]
    }
}

impl<T: Float, D: Device<T>> Operation<T, D> for Cos {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].cos()
    }

    fn gradient(&self, out_grad: &Tensor<T, D>, node: &Tensor<T, D>) -> Vec<Tensor<T, D>> {
        vec![out_grad * &-&node.0.read().unwrap().inputs[0].sin()]
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

fn reduce_to_shape<T: Type, D: Device<T>>(
    grads: Vec<Tensor<T, D>>,
    inputs: &[Tensor<T, D>],
) -> Vec<Tensor<T, D>> {
    grads
        .iter()
        .zip(inputs)
        .map(|(grad, input)| {
            let input_shape = input.shape();
            if input_shape != grad.shape() {
                reduce_by_add(grad, &input_shape)
            } else {
                grad.clone()
            }
        })
        .collect()
}

pub fn reduce_by_add<T: Type, D: Device<T>>(
    input: &Tensor<T, D>,
    output_shape: &[usize],
) -> Tensor<T, D> {
    let input_shape = input.shape();
    let n = input_shape.len() - output_shape.len();
    let sum = if n > 0 {
        input.sum(Some((0..n).collect::<Vec<_>>()), false)
    } else {
        input.clone()
    };
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
