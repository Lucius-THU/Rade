use crate::device::Device;
use crate::ndarray::NDArray;
use crate::tensor::Tensor;
use crate::type_trait::{PowTensor, Type};
use num_traits::{Float, One, Pow};

pub(crate) trait Operation<'a, T: Type + 'a, D: Device> {
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

pub(crate) struct PowScalar<T: Type>(pub T);

pub(crate) struct ScalarPow<T: Type>(pub T);

pub(crate) struct Ln;

impl<'a, T: Type + 'a, D: Device> Operation<'a, T, D> for Broadcast {
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

impl<'a, T: Type + 'a, D: Device> Operation<'a, T, D> for Summation {
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

impl<'a, T: Type + 'a, D: Device> Operation<'a, T, D> for Reshape {
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

impl<'a, T: Type + 'a, D: Device> Operation<'a, T, D> for EWiseAdd {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs + rhs)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let inputs = &node.0.write().unwrap().inputs;
        let in_grads = vec![out_grad.clone(), out_grad.clone()];
        reduce_to_shape(in_grads, inputs, out_grad.shape())
    }
}

impl<'a, T: Type + 'a, D: Device> Operation<'a, T, D> for AddScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        &args[0] + self.0
    }

    fn gradient(&self, out_grad: &Tensor<'a, T, D>, _: &Tensor<'a, T, D>) -> Vec<Tensor<'a, T, D>> {
        vec![out_grad.clone()]
    }
}

impl<'a, T: Type + 'a, D: Device> Operation<'a, T, D> for EWiseMul {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs * rhs)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let inputs = &node.0.write().unwrap().inputs;
        let in_grads = vec![out_grad * &inputs[1], out_grad * &inputs[0]];
        reduce_to_shape(in_grads, inputs, out_grad.shape())
    }
}

impl<'a, T: Type + 'a, D: Device> Operation<'a, T, D> for MulScalar<T> {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        &args[0] * self.0
    }

    fn gradient(&self, out_grad: &Tensor<'a, T, D>, _: &Tensor<'a, T, D>) -> Vec<Tensor<'a, T, D>> {
        vec![out_grad * self.0]
    }
}

impl<'a, T: Type + Float + Pow<T, Output = T> + 'a, D: Device> Operation<'a, T, D> for EWisePow {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        apply_with_broadcast(args, |lhs, rhs| lhs.pow(rhs))
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let inputs = &node.0.write().unwrap().inputs;
        let in_grads = vec![
            &(out_grad * &inputs[1]) * &inputs[0].pow(&(&inputs[1] - T::one())),
            &(out_grad * &inputs[0].pow(&inputs[1])) * &inputs[1].ln(),
        ];
        reduce_to_shape(in_grads, inputs, out_grad.shape())
    }
}

impl<'a, D: Device, T: Type + Float + Pow<T, Output = T> + PowTensor + 'a> Operation<'a, T, D>
    for ScalarPow<T>
{
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].scalar_pow(self.0)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let input = &node.0.write().unwrap().inputs[0];
        vec![&(out_grad * &self.0.powt(input)) * self.0.ln()]
    }
}

impl<'a, T: Type + Float + Pow<T, Output = T> + 'a, D: Device> Operation<'a, T, D>
    for PowScalar<T>
{
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].pow(self.0)
    }

    fn gradient(
        &self,
        out_grad: &Tensor<'a, T, D>,
        node: &Tensor<'a, T, D>,
    ) -> Vec<Tensor<'a, T, D>> {
        let input = &node.0.write().unwrap().inputs[0];
        vec![&(out_grad * self.0) * &input.pow(self.0 - T::one())]
    }
}

macro_rules! impl_pow_scalar {
    ($t:ty, $u:ty) => {
        impl<'a, D: Device> Operation<'a, $t, D> for PowScalar<$u> {
            fn compute(&self, args: &[NDArray<$t, D>]) -> NDArray<$t, D> {
                args[0].pow(self.0)
            }

            fn gradient(
                &self,
                out_grad: &Tensor<'a, $t, D>,
                node: &Tensor<'a, $t, D>,
            ) -> Vec<Tensor<'a, $t, D>> {
                let input = &node.0.write().unwrap().inputs[0];
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
impl_pow_scalar!(i128, u32);

impl<'a, T: Type + Float + Pow<T, Output = T> + 'a, D: Device> Operation<'a, T, D> for Ln {
    fn compute(&self, args: &[NDArray<T, D>]) -> NDArray<T, D> {
        args[0].ln()
    }

    fn gradient(&self, out_grad: &Tensor<'a, T, D>, _: &Tensor<'a, T, D>) -> Vec<Tensor<'a, T, D>> {
        vec![out_grad / &out_grad.0.write().unwrap().inputs[0]]
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

fn reduce_to_shape<'a, T: Type, D: Device>(
    mut grads: Vec<Tensor<'a, T, D>>,
    inputs: &[Tensor<T, D>],
    shape: Vec<usize>,
) -> Vec<Tensor<'a, T, D>> {
    for i in 0..inputs.len() {
        let input_shape = inputs[i].shape();
        if input_shape != shape {
            grads[i] = reduce_by_add(&grads[i], &input_shape);
        }
    }
    grads
}

fn reduce_by_add<'a, T: Type, D: Device>(
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

fn apply_with_broadcast<T: Type, D: Device>(
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
