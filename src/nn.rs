use crate::device::Device;
use crate::tensor::Tensor;
use crate::type_trait::{Float, Type};
use num_traits::Pow;

pub trait Module<'a, T: Type + 'a, D: Device> {
    fn forward(&mut self, input: &Tensor<'a, T, D>) -> Tensor<'a, T, D>;
    fn parameters(&self) -> Vec<&Tensor<'a, T, D>>;
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct Linear<'a, T: Float, D: Device> {
    weight: Tensor<'a, T, D>,
    bias: Option<Tensor<'a, T, D>>,
}

pub struct ReLU;

pub struct Sequential<'a, T: Type, D: Device>(Vec<Box<dyn Module<'a, T, D> + 'a>>);

pub struct BatchNorm<'a, T: Float, D: Device> {
    weight: Tensor<'a, T, D>,
    bias: Tensor<'a, T, D>,
    running_mean: Tensor<'a, T, D>,
    running_var: Tensor<'a, T, D>,
    eps: T,
    momentum: T,
    training: bool,
}

pub struct Dropout<T: Float> {
    p: T,
    training: bool,
}

pub struct Residual<'a, T: Type, D: Device>(Box<dyn Module<'a, T, D>>);

impl<'a, T: Float, D: Device> Linear<'a, T, D> {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Tensor::kaiming_uniform(in_features, out_features, true);
        let bias = if bias {
            Some(Tensor::kaiming_uniform(out_features, 1, true))
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl<'a, T: Float, D: Device> Module<'a, T, D> for Linear<'a, T, D> {
    fn forward(&mut self, input: &Tensor<'a, T, D>) -> Tensor<'a, T, D> {
        let mut output = input.matmul(&self.weight);
        if let Some(bias) = &self.bias {
            output = &output + bias;
        }
        output
    }

    fn parameters(&self) -> Vec<&Tensor<'a, T, D>> {
        let mut parameters = vec![&self.weight];
        if let Some(bias) = &self.bias {
            parameters.push(bias);
        }
        parameters
    }
}

impl<'a, T: Type + 'a, D: Device> Module<'a, T, D> for ReLU {
    fn forward(&mut self, input: &Tensor<'a, T, D>) -> Tensor<'a, T, D> {
        input.relu()
    }

    fn parameters(&self) -> Vec<&Tensor<'a, T, D>> {
        vec![]
    }
}

impl<'a, T: Type, D: Device> Sequential<'a, T, D> {
    pub fn new(modules: Vec<Box<dyn Module<'a, T, D> + 'a>>) -> Self {
        Self(modules)
    }
}

impl<'a, T: Type, D: Device> Module<'a, T, D> for Sequential<'a, T, D> {
    fn forward(&mut self, input: &Tensor<'a, T, D>) -> Tensor<'a, T, D> {
        let mut output = input.clone();
        for module in &mut self.0 {
            output = module.forward(&output);
        }
        output
    }

    fn parameters(&self) -> Vec<&Tensor<'a, T, D>> {
        let mut parameters = Vec::new();
        for module in &self.0 {
            parameters.extend(module.parameters());
        }
        parameters
    }

    fn train(&mut self) {
        for module in &mut self.0 {
            module.train()
        }
    }

    fn eval(&mut self) {
        for module in &mut self.0 {
            module.eval()
        }
    }
}

impl<'a, T: Float, D: Device> BatchNorm<'a, T, D> {
    pub fn new(dim: usize, eps: T, momentum: T) -> Self {
        BatchNorm {
            weight: Tensor::ones(&[dim], true),
            bias: Tensor::zeros(&[dim], true),
            running_mean: Tensor::zeros(&[dim], false),
            running_var: Tensor::ones(&[dim], false),
            eps,
            momentum,
            training: true,
        }
    }
}

impl<'a, T: Float, D: Device> Module<'a, T, D> for BatchNorm<'a, T, D> {
    fn forward(&mut self, input: &Tensor<'a, T, D>) -> Tensor<'a, T, D> {
        if !self.training {
            &(&(&self.weight * &(input - &self.running_mean))
                / &(&self.running_var + self.eps).pow(T::from(0.5).unwrap()))
                + &self.bias
        } else {
            let batch = T::from(input.shape()[0]).unwrap();
            let e_x = &input.sum(Some(vec![0]), false) / batch;
            let var_x = &(input - &e_x)
                .pow(T::from(2).unwrap())
                .sum(Some(vec![0]), false)
                / batch;
            self.running_mean =
                &(&e_x * self.momentum) + &(&self.running_mean * (T::one() - self.momentum));
            self.running_var =
                &(&var_x * self.momentum) + &(&self.running_var * (T::one() - self.momentum));
            &(&(&self.weight * &(input - &e_x)) / &(&var_x + self.eps).pow(T::from(0.5).unwrap()))
                + &self.bias
        }
    }

    fn parameters(&self) -> Vec<&Tensor<'a, T, D>> {
        vec![&self.weight, &self.bias]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

impl<T: Float> Dropout<T> {
    pub fn new(p: T) -> Self {
        Dropout { p, training: true }
    }
}

impl<'a, T: Float + 'a, D: Device> Module<'a, T, D> for Dropout<T> {
    fn forward(&mut self, input: &Tensor<'a, T, D>) -> Tensor<'a, T, D> {
        if !self.training {
            input.clone()
        } else {
            &(input * &Tensor::rand(&input.shape(), T::zero(), T::one(), false).gt(self.p))
                / (T::one() - self.p)
        }
    }

    fn parameters(&self) -> Vec<&Tensor<'a, T, D>> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

impl<'a, T: Type, D: Device> Residual<'a, T, D> {
    pub fn new(module: Box<dyn Module<'a, T, D>>) -> Self {
        Self(module)
    }
}

impl<'a, T: Type, D: Device> Module<'a, T, D> for Residual<'a, T, D> {
    fn forward(&mut self, input: &Tensor<'a, T, D>) -> Tensor<'a, T, D> {
        input + &self.0.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor<'a, T, D>> {
        vec![]
    }

    fn train(&mut self) {
        self.0.train()
    }

    fn eval(&mut self) {
        self.0.eval()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CPU;

    #[test]
    fn test_linear() {
        let mut linear = Linear::<f32, CPU>::new(3, 2, true);
        let input = Tensor::ones(&[2, 3], false);
        let output = linear.forward(&input);
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(linear.parameters().len(), 2);
    }

    #[test]
    fn test_sequential() {
        let linear1 = Linear::<f32, CPU>::new(3, 2, true);
        let linear2 = Linear::new(2, 1, true);
        let mut sequential =
            Sequential::new(vec![Box::new(linear1), Box::new(ReLU), Box::new(linear2)]);
        let input = Tensor::ones(&[2, 3], false);
        let output = sequential.forward(&input);
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(sequential.parameters().len(), 4);
        println!("{output}");
    }
}