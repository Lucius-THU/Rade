use crate::device::Device;
use crate::nn::{Embedding, Linear, Module, RotaryEmbedding, Sequential, TransformerBlock};
use crate::tensor::Tensor;
use crate::type_trait::{Float, Unsigned};
use rand_distr::{Distribution, StandardNormal};

pub struct Llama2<T: Float, D: Device<T>> {
    embedding: Embedding<T, D>,
    sequential: Sequential<T, D>,
    linear: Linear<T, D>,
}

impl<T: Float, D: Device<T> + 'static> Llama2<T, D>
where
    StandardNormal: Distribution<T>,
{
    pub fn new(
        n_voc: usize,
        hidden_dim: usize,
        n_heads: usize,
        intermediate_size: usize,
        n_layers: usize,
        max_seq_len: usize,
        rotary_base: T,
    ) -> Self {
        let rotatory_emb = RotaryEmbedding::new(hidden_dim / n_heads, max_seq_len, rotary_base);
        let mut blocks: Vec<Box<dyn Module<T, T, D>>> = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(Box::new(TransformerBlock::new(
                hidden_dim,
                n_heads,
                intermediate_size,
                true,
                T::from(0.1).unwrap(),
                &rotatory_emb,
            )));
        }
        Self {
            embedding: Embedding::new(n_voc, hidden_dim),
            sequential: Sequential::new(blocks),
            linear: Linear::new(hidden_dim, n_voc, false),
        }
    }
}

impl<T: Float, U: Unsigned, D: Device<T> + Device<U> + 'static> Module<T, U, D> for Llama2<T, D> {
    fn forward(&mut self, input: &Tensor<U, D>) -> Tensor<T, D> {
        let x = self.embedding.forward(input);
        let x = self.sequential.forward(&x);
        self.linear.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor<T, D>> {
        let mut params = <Embedding<T, D> as Module<T, U, D>>::parameters(&self.embedding);
        params.extend(self.sequential.parameters());
        params.extend(self.linear.parameters());
        params
    }

    fn state_dict(&self) -> Vec<Tensor<T, D>> {
        let mut params = <Embedding<T, D> as Module<T, U, D>>::state_dict(&self.embedding);
        params.extend(self.sequential.state_dict());
        params.extend(self.linear.state_dict());
        params
    }

    fn train(&mut self) {
        self.sequential.train();
    }

    fn eval(&mut self) {
        self.sequential.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CPU;
    use crate::functional::cross_entropy_loss;

    #[test]
    fn test_llama() {
        let mut model = Llama2::<f32, CPU>::new(100, 128, 4, 512, 3, 8, 10000.);
        let input = Tensor::<usize, CPU>::new1d([0, 72, 36, 8, 7, 1, 13, 24], false);
        let output = model.forward(&input);
        let loss = cross_entropy_loss(&output.split(0, 0, 7), &input.split(0, 1, 7));
        loss.backward();
    }
}
