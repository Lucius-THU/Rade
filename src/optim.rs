use crate::device::Device;
use crate::tensor::Tensor;
use crate::type_trait::{Float, Type};
use num_traits::Pow;

pub trait Optimizer<T: Type, D: Device> {
    fn parameters(&self) -> &[Tensor<T, D>];
    fn step(&mut self);
    fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.zero_grad();
        }
    }
}

pub struct SGD<T: Type, D: Device> {
    parameters: Vec<Tensor<T, D>>,
    lr: T,
    momentum: T,
    weight_decay: T,
    u: Vec<Tensor<T, D>>,
}

pub struct Adam<T: Float, D: Device> {
    parameters: Vec<Tensor<T, D>>,
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    u: Vec<Tensor<T, D>>,
    v: Vec<Tensor<T, D>>,
    t: usize,
}

impl<T: Type, D: Device> SGD<T, D> {
    pub fn new(parameters: Vec<Tensor<T, D>>, lr: T, momentum: T, weight_decay: T) -> Self {
        let len = parameters.len();
        Self {
            parameters,
            lr,
            momentum,
            weight_decay,
            u: Vec::with_capacity(len),
        }
    }
}

impl<T: Type, D: Device> Optimizer<T, D> for SGD<T, D> {
    fn parameters(&self) -> &[Tensor<T, D>] {
        &self.parameters
    }

    fn step(&mut self) {
        for (i, parameter) in self.parameters.iter().enumerate() {
            let grad = (&(&parameter.grad().unwrap() + &(parameter * self.weight_decay))
                * (T::one() - self.momentum))
                .detach(false);
            if self.u.len() <= i {
                self.u.push(grad);
            } else {
                self.u[i] = (&(&self.u[i] * self.momentum) + &grad).detach(false);
            }
            parameter.set_data(parameter - &(&self.u[i] * self.lr));
        }
    }
}

impl<T: Float, D: Device> Adam<T, D> {
    pub fn new(
        parameters: Vec<Tensor<T, D>>,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
    ) -> Self {
        let len = parameters.len();
        Self {
            parameters,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            u: Vec::with_capacity(len),
            v: Vec::with_capacity(len),
            t: 0,
        }
    }
}

impl<'a, T: Float, D: Device> Optimizer<T, D> for Adam<T, D> {
    fn parameters(&self) -> &[Tensor<T, D>] {
        &self.parameters
    }

    fn step(&mut self) {
        self.t += 1;
        for (i, parameter) in self.parameters.iter().enumerate() {
            let grad =
                (&parameter.grad().unwrap() + &(parameter * self.weight_decay)).detach(false);
            if self.u.len() <= i {
                self.u.push((&grad * (T::one() - self.beta1)).detach(false));
                self.v.push(
                    (&(&grad * &grad) * (T::one() - self.beta2)).detach(false),
                );
            } else {
                self.u[i] =
                    (&(&self.u[i] * self.beta1) + &(&grad * (T::one() - self.beta1))).detach(false);
                self.v[i] = (&(&self.v[i] * self.beta2)
                    + &(&(&grad * &grad) * (T::one() - self.beta2)))
                    .detach(false);
            }
            let u = &self.u[i] / (T::one() - self.beta1.pow(T::from(self.t).unwrap()));
            let v = &self.v[i] / (T::one() - self.beta2.pow(T::from(self.t).unwrap()));
            parameter.set_data(
                parameter - &(&(&u * self.lr) / &(&v.pow(T::from(0.5).unwrap()) + self.eps)),
            );
        }
    }
}
