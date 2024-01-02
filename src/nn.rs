pub mod model;

use crate::device::Device;
use crate::functional::softmax;
use crate::tensor::Tensor;
use crate::type_trait::{Float, Type, Unsigned};
use bincode::config;
use bincode::error::{DecodeError, EncodeError};
use rand_distr::{Distribution, StandardNormal};
use std::{fs, path::Path};

pub trait Module<T: Type, U: Type, D: Device<T> + Device<U>> {
    fn forward(&mut self, input: &Tensor<U, D>) -> Tensor<T, D>;
    fn parameters(&self) -> Vec<Tensor<T, D>>;
    fn state_dict(&self) -> Vec<Tensor<T, D>>;
    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn save(&self, path: &str) -> Result<(), EncodeError> {
        let msg = "File cannot be created.";
        if let Some(p) = Path::new(path).parent() {
            if let Err(_) = fs::create_dir_all(p) {
                Err(EncodeError::Other(msg))
            } else {
                if let Ok(mut file) = fs::File::create(path) {
                    for parameter in self.state_dict() {
                        bincode::encode_into_std_write(parameter, &mut file, config::standard())?;
                    }
                    Ok(())
                } else {
                    Err(EncodeError::Other(msg))
                }
            }
        } else {
            Err(EncodeError::OtherString(format!(
                "{} is not a valid path.",
                path
            )))
        }
    }

    fn load(&mut self, path: &str) -> Result<(), DecodeError> {
        if let Ok(mut file) = fs::File::open(path) {
            for parameter in self.state_dict() {
                parameter.set_data(bincode::decode_from_std_read(
                    &mut file,
                    config::standard(),
                )?);
            }
            self.eval();
            Ok(())
        } else {
            Err(DecodeError::Other("File not found."))
        }
    }
}

pub struct Linear<T: Float, D: Device<T>> {
    weight: Tensor<T, D>,
    bias: Option<Tensor<T, D>>,
}

pub struct ReLU;

pub struct Sequential<T: Type, D: Device<T>>(Vec<Box<dyn Module<T, T, D>>>);

pub struct BatchNorm<T: Float, D: Device<T>> {
    weight: Tensor<T, D>,
    bias: Tensor<T, D>,
    running_mean: Tensor<T, D>,
    running_var: Tensor<T, D>,
    eps: T,
    momentum: T,
    training: bool,
}

pub struct RMSNorm<T: Float, D: Device<T>> {
    weight: Tensor<T, D>,
    eps: T,
}

pub struct Dropout<T: Float> {
    p: T,
    training: bool,
}

pub struct Residual<T: Type, D: Device<T>>(Box<dyn Module<T, T, D>>);

pub struct Embedding<T: Type, D: Device<T>>(Tensor<T, D>);

pub struct Attention<T: Float, D: Device<T>> {
    w_q: Linear<T, D>,
    w_k: Linear<T, D>,
    w_v: Linear<T, D>,
    w_o: Linear<T, D>,
    dropout: Dropout<T>,
    rotary_embedding: RotaryEmbedding<T, D>,
    n_heads: usize,
    head_dim: usize,
    mask: Option<Tensor<T, D>>
}

pub struct SwishFFN<T: Float, D: Device<T>> {
    up: Linear<T, D>,
    gate: Linear<T, D>,
    down: Linear<T, D>,
}

pub struct TransformerBlock<T: Float, D: Device<T>> {
    sequential: Sequential<T, D>,
}

#[derive(Clone)]
pub struct RotaryEmbedding<T: Float, D: Device<T>> {
    sin: Tensor<T, D>,
    cos: Tensor<T, D>,
}

impl<T: Float, D: Device<T>> Linear<T, D> {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Tensor::kaiming_uniform(in_features, out_features, true);
        let bias = if bias {
            Some(
                Tensor::kaiming_uniform(out_features, 1, true)
                    .reshape(vec![out_features])
                    .detach(true),
            )
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl<T: Float, D: Device<T>> Module<T, T, D> for Linear<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        let mut output = input.matmul(&self.weight);
        if let Some(bias) = &self.bias {
            output = &output + bias;
        }
        output
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        let mut parameters = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            parameters.push(bias.clone());
        }
        parameters
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        self.parameters()
    }
}

impl<T: Type, D: Device<T>> Module<T, T, D> for ReLU {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        vec![]
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        vec![]
    }
}

impl<T: Type, D: Device<T>> Sequential<T, D> {
    pub fn new(modules: Vec<Box<dyn Module<T, T, D>>>) -> Self {
        Self(modules)
    }
}

impl<T: Type, D: Device<T>> Module<T, T, D> for Sequential<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        let mut output = input.clone();
        for module in &mut self.0 {
            output = module.forward(&output);
        }
        output
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        let mut parameters = Vec::new();
        for module in &self.0 {
            parameters.extend(module.parameters());
        }
        parameters
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        let mut parameters = Vec::new();
        for module in &self.0 {
            parameters.extend(module.state_dict());
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

impl<T: Float, D: Device<T>> BatchNorm<T, D> {
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

impl<T: Float, D: Device<T>> Module<T, T, D> for BatchNorm<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        if !self.training {
            &(&(&self.weight * &(input - &self.running_mean))
                / &(&self.running_var + self.eps).sqrt())
                + &self.bias
        } else {
            let batch = T::from(input.shape()[0]).unwrap();
            let e_x = &input.sum(Some(vec![0]), false) / batch;
            let diff_x = &(input - &e_x);
            let var_x = &(diff_x * diff_x).sum(Some(vec![0]), false) / batch;
            self.running_mean = (&(&e_x * self.momentum)
                + &(&self.running_mean * (T::one() - self.momentum)))
                .detach(false);
            self.running_var = (&(&var_x * self.momentum)
                + &(&self.running_var * (T::one() - self.momentum)))
                .detach(false);
            &(&(&self.weight * &(input - &e_x)) / &(&var_x + self.eps).sqrt()) + &self.bias
        }
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        vec![
            self.weight.clone(),
            self.bias.clone(),
            self.running_mean.clone(),
            self.running_var.clone(),
        ]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

impl<T: Float, D: Device<T>> RMSNorm<T, D> {
    pub fn new(dim: usize, eps: T) -> Self {
        RMSNorm {
            weight: Tensor::ones(&[dim], true),
            eps,
        }
    }
}

impl<T: Float, D: Device<T>> Module<T, T, D> for RMSNorm<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        &(input
            / &(&(input * input)
                .sum(Some(vec![input.ndim() - 1]), true)
                .sqrt()
                + self.eps))
            * &self.weight
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        vec![self.weight.clone()]
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        vec![self.weight.clone()]
    }
}

impl<T: Float> Dropout<T> {
    pub fn new(p: T) -> Self {
        Dropout { p, training: true }
    }
}

impl<T: Float, D: Device<T>> Module<T, T, D> for Dropout<T> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        if !self.training {
            input.clone()
        } else {
            &(input * &Tensor::rand(&input.shape(), T::zero(), T::one(), false).gt(self.p))
                / (T::one() - self.p)
        }
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        vec![]
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

impl<T: Type, D: Device<T>> Residual<T, D> {
    pub fn new(module: Box<dyn Module<T, T, D>>) -> Self {
        Self(module)
    }
}

impl<T: Type, D: Device<T>> Module<T, T, D> for Residual<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        input + &self.0.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        self.0.parameters()
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        self.0.state_dict()
    }

    fn train(&mut self) {
        self.0.train()
    }

    fn eval(&mut self) {
        self.0.eval()
    }
}

impl<T: Float, D: Device<T>> Embedding<T, D>
where
    StandardNormal: Distribution<T>,
{
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self(Tensor::randn(
            &[num_embeddings, embedding_dim],
            T::zero(),
            T::one(),
            true,
        ))
    }
}

impl<T: Float, U: Unsigned, D: Device<T> + Device<U> + 'static> Module<T, U, D>
    for Embedding<T, D>
{
    fn forward(&mut self, input: &Tensor<U, D>) -> Tensor<T, D> {
        self.0.index(input)
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        vec![self.0.clone()]
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        vec![self.0.clone()]
    }
}

impl<T: Float, D: Device<T>> Attention<T, D> {
    pub fn new(
        dim: usize,
        n_heads: usize,
        mask: Option<Tensor<T, D>>,
        p: T,
        rotary_embedding: &RotaryEmbedding<T, D>,
    ) -> Self {
        let head_dim = dim / n_heads;
        if head_dim != rotary_embedding.sin.shape()[1] {
            panic!("The dimension of the rotary embedding must be equal to the dimension of the hidden state divided by the number of heads.");
        }
        if let Some(mask) = &mask {
            if mask.shape()[0] != mask.shape()[1] {
                panic!("The shape of the mask must be square.");
            }
        }

        let hidden_dim = head_dim * n_heads;
        let w_q = Linear::new(dim, hidden_dim, false);
        let w_k = Linear::new(dim, hidden_dim, false);
        let w_v = Linear::new(dim, hidden_dim, false);
        let w_o = Linear::new(dim, hidden_dim, false);
        let dropout = Dropout::new(p);
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            dropout,
            rotary_embedding: rotary_embedding.clone(),
            n_heads,
            head_dim,
            mask,
        }
    }
}

impl<T: Float, D: Device<T>> Module<T, T, D> for Attention<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        let mut shape = input.shape();
        let len = shape.len();
        let scale = T::from(self.head_dim).unwrap().sqrt();
        let seq_len = shape[len - 2];
        if let Some(mask) = &self.mask {
            if mask.shape()[0] != seq_len {
                panic!("The shape of the mask must be equal to the shape of the input.");
            }
        }
        shape[len - 1] = self.n_heads;
        shape.push(self.head_dim);
        let q = self
            .w_q
            .forward(input)
            .reshape(shape.clone())
            .transpose(Some((len - 2, len - 1)));
        let k = self
            .w_k
            .forward(input)
            .reshape(shape.clone())
            .transpose(Some((len - 2, len - 1)));
        let v = self
            .w_v
            .forward(input)
            .reshape(shape.clone())
            .transpose(Some((len - 2, len - 1)));
        let (sin, cos) = self.rotary_embedding.forward(seq_len);
        let q_embed = &(&(&q * &cos) + &(&rotate_half(&q) * &sin)) / scale;
        let k_embed = &(&k * &cos) + &(&rotate_half(&k) * &sin);
        let mut q_k = q_embed.matmul(&k_embed.transpose(Some((len - 1, len))));
        if let Some(mask) = self.mask.as_ref() {
            q_k = &q_k + mask;
        }
        self.w_o.forward(
            &self
                .dropout
                .forward(&softmax(&q_k, len))
                .matmul(&v)
                .transpose(Some((len - 2, len - 1)))
                .reshape(input.shape()),
        )
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        vec![
            self.w_q.weight.clone(),
            self.w_k.weight.clone(),
            self.w_v.weight.clone(),
            self.w_o.weight.clone(),
        ]
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        vec![
            self.w_q.weight.clone(),
            self.w_k.weight.clone(),
            self.w_v.weight.clone(),
            self.w_o.weight.clone(),
        ]
    }

    fn train(&mut self) {
        <Dropout<T> as Module<T, T, D>>::train(&mut self.dropout);
    }

    fn eval(&mut self) {
        <Dropout<T> as Module<T, T, D>>::eval(&mut self.dropout);
    }
}

impl<T: Float, D: Device<T>> SwishFFN<T, D> {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        Self {
            up: Linear::new(dim, hidden_dim, false),
            gate: Linear::new(dim, hidden_dim, false),
            down: Linear::new(hidden_dim, dim, false),
        }
    }
}

impl<T: Float, D: Device<T>> Module<T, T, D> for SwishFFN<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        let gate = self.gate.forward(input);
        let act = &gate / &(&(-&gate).exp() + T::one());
        self.down.forward(&(&self.up.forward(input) * &act))
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        vec![
            self.up.weight.clone(),
            self.gate.weight.clone(),
            self.down.weight.clone(),
        ]
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        vec![
            self.up.weight.clone(),
            self.gate.weight.clone(),
            self.down.weight.clone(),
        ]
    }
}

impl<T: Float, D: Device<T> + 'static> TransformerBlock<T, D> {
    pub fn new(
        dim: usize,
        n_heads: usize,
        hidden_dim: usize,
        mask: Option<Tensor<T, D>>,
        p: T,
        rotary_embedding: &RotaryEmbedding<T, D>,
    ) -> Self {
        Self {
            sequential: Sequential::new(vec![
                Box::new(Residual::new(Box::new(Sequential::new(vec![
                    Box::new(RMSNorm::new(dim, T::from(1e-5).unwrap())),
                    Box::new(Attention::new(dim, n_heads, mask, p, rotary_embedding)),
                ])))),
                Box::new(Residual::new(Box::new(Sequential::new(vec![
                    Box::new(RMSNorm::new(dim, T::from(1e-5).unwrap())),
                    Box::new(SwishFFN::new(dim, hidden_dim)),
                ])))),
            ]),
        }
    }
}

impl<T: Float, D: Device<T>> Module<T, T, D> for TransformerBlock<T, D> {
    fn forward(&mut self, input: &Tensor<T, D>) -> Tensor<T, D> {
        self.sequential.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        self.sequential.parameters()
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        self.sequential.state_dict()
    }

    fn train(&mut self) {
        self.sequential.train()
    }

    fn eval(&mut self) {
        self.sequential.eval()
    }
}

impl<T: Float, D: Device<T>> RotaryEmbedding<T, D> {
    pub fn new(dim: usize, seq_len: usize, base: T) -> Self {
        let half_dim = T::from(dim / 2).unwrap();
        let freq = base.powt(&(&Tensor::arange(T::zero(), half_dim, T::one(), false) / half_dim));
        let inv_freq = &Tensor::ones_like(&freq, false) / &freq;
        let t = Tensor::arange(T::zero(), T::from(seq_len).unwrap(), T::one(), false);
        let t_freq = t
            .reshape(vec![seq_len, 1])
            .matmul(&inv_freq.reshape(vec![1, dim / 2]));
        let emb = t_freq.cat(&[&t_freq], 1);
        Self {
            sin: emb.sin(),
            cos: emb.cos(),
        }
    }

    pub fn forward(&self, seq_len: usize) -> (Tensor<T, D>, Tensor<T, D>) {
        (self.sin.split(0, 0, seq_len), self.cos.split(0, 0, seq_len))
    }
}

fn rotate_half<T: Type, D: Device<T>>(hidden_state: &Tensor<T, D>) -> Tensor<T, D> {
    let dim = hidden_state.ndim() - 1;
    let len = hidden_state.shape()[dim] / 2;
    (-&hidden_state.split(dim, len, len)).cat(&[&hidden_state.split(dim, 0, len)], dim)
}

#[cfg(test)]
mod tests {
    use num_traits::Float;
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
    }

    #[test]
    fn test_transformer_block() {
        let mut mask = Vec::with_capacity(16);
        for i in 0..4 {
            for j in 0..4 {
                if i < j {
                    mask.push(f32::neg_infinity());
                } else {
                    mask.push(0.);
                }
            }
        }
        let mask = Tensor::new_with_shape(&mask, &[4, 4], false);
        let mut transformer_block = TransformerBlock::<f32, CPU>::new(
            4,
            2,
            8,
            Some(mask.clone()),
            0.1,
            &RotaryEmbedding::new(2, 4, 10000.),
        );
        let input = Tensor::ones(&[2, 4, 4], true);
        let output = transformer_block.forward(&input);
        output.backward();
        assert_eq!(output.shape(), &[2, 4, 4]);
    }
}
