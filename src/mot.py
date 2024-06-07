import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from config import BATCH_SIZE, INPUT_SIZE, DEPTH, NB_NEURONS, L_R, MAX_ITER

def convex_biconjugate(f, x_vals: np.ndarray, min_slope: float=-1., max_slope: float=1., n_steps: int=1000):
    # Range of possible slopes
    m_vals = np.linspace(min_slope, max_slope, n_steps)

    def f_star(y):
        return np.max(y * x_vals - f(x_vals), axis=0)

    def f_bistar(x):
        return np.max(m_vals * x - f_star(m_vals), axis=1)

    return f_bistar

def payoff(x, y):
    return jnp.maximum(y - x, 0)

def sample(cdf, xaxis, n_samples: int):
    u = np.random.uniform(0, 1, n_samples)
    return np.interp(u, cdf, xaxis)
        
# We need to sample values where to evaluate the convex enveloppe
def sample_xaxis(n_samples):
    u = np.random.uniform(0, 1, n_samples)
    return np.interp(u, short_cdf, xs)

class MLP(nn.Module):
    nb_neurons: int
    depth: int

    def setup(self):
        self.dense_layers = [nn.Dense(self.nb_neurons // (2 ** i)) for i in range(self.depth)]
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.nb_neurons)(x)
        for layer in self.dense_layers:
            x = layer(x)
        return nn.Dense(1)(x)
    
def loss_fn(params, x_1, x_2):
    batch_size = x_1.shape[0]
    x_vals = sample_xaxis(batch_size)
    
    def c(x, y):
        cost = payoff(x, y)
        u = model.apply(params, y)
        return cost - u
    
    res = convex_biconjugate(lambda y: c(x_1, y), x_vals, n_steps=batch_size)(np.reshape(x_1,(batch_size,1)))
    u = model.apply(params, x_2)
    
    loss = res + u
    return jnp.mean(loss)

@jax.jit
def value_and_grad_fn(params, x_1, x_2):
    return loss_fn(params, x_1, x_2), jax.grad(loss_fn)(params, x_1, x_2)


if __name__ == "__main__":
    short_cdf, long_cdf, xs = np.load("data/short_cdf.npy"), np.load("data/long_cdf.npy"), np.load("data/xs.npy")

    # Init model
    model = MLP(nb_neurons=NB_NEURONS, depth=DEPTH)
    rng = jax.random.PRNGKey(0)
    input_shape = (BATCH_SIZE, INPUT_SIZE)
    params = model.init(rng, jnp.ones(input_shape))
    # Adam as an optimizer, init
    tx = optax.adam(learning_rate=L_R)
    opt_state = tx.init(params)

    for epoch in range(MAX_ITER):
        x1_batch = sample(cdf=short_cdf, xaxis=xs, n_samples=BATCH_SIZE)
        x2_batch = sample(cdf=long_cdf, xaxis=xs, n_samples=BATCH_SIZE)

        loss_val, grads = value_and_grad_fn(params, x1_batch, x2_batch)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if epoch % 100 == 0 or epoch < 5 and epoch > 0:
            print(f"Iteration: {epoch}, Avg. Loss: {loss_val}")




