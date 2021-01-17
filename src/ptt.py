
from collections import namedtuple
import numpy as np
import h5py
import yaml
import sys
import timeit
from typing import Callable

# need 64 bit floats to compute cumsum differences to do the inverse ptt
from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

PttArgs = namedtuple("PTTArgs", ["leaf_permutation", "max_leaf", "min_leaf", "left_min_leaf"])


"""
Construct necessary parameters to compute the inverse ptt.
"""
def inverse_ptt_params(left_index, right_index, leaf_index):
    num_nodes = len(left_index)
    n = (num_nodes+1)//2

    # NOTE: keep in mind, are serialized in dfs order, but for whatever reason
    # visiting the right node first, so this whole thing is backwards from what
    # you might expect.

    # compute leaf permutation
    leaf_permutation = np.zeros(n, np.int32)
    min_leaf_index = np.zeros([num_nodes], np.int)
    max_leaf_index = np.zeros([num_nodes], np.int)
    k = 0 # leaf node number
    for i in range(num_nodes):
        if leaf_index[i] >= 0:
            leaf_permutation[k] = leaf_index[i]
            min_leaf_index[i] = k
            max_leaf_index[i] = k
            k += 1
    assert k == n

    # figure out subtree spans for every node
    for i in range(num_nodes-1, -1, -1):
        if leaf_index[i] < 0:
            min_leaf_index[i] = min_leaf_index[right_index[i]]
            max_leaf_index[i] = max_leaf_index[left_index[i]]
            assert min_leaf_index[i] < max_leaf_index[i]

    # now just compute the indexes we need to compute values for internal nodes
    max_leaf = np.zeros(n-1, np.int32)
    min_leaf = np.zeros(n-1, np.int32)
    left_min_leaf = np.zeros(n-1, np.int32)

    k = 0 # internal node number
    for i in range(num_nodes):
        if leaf_index[i] >= 0:
            continue
        max_leaf[k] = max_leaf_index[i] + 1
        min_leaf[k] = min_leaf_index[i]
        left_min_leaf[k] = min_leaf_index[left_index[i]]
        k += 1

    return PttArgs(
        jax.device_put(leaf_permutation),
        jax.device_put(max_leaf),
        jax.device_put(min_leaf),
        jax.device_put(left_min_leaf))


"""
Compute the inverse Polya tree transformation y = T^{-1}(x) where
x is in a n-1 dimensional simplex, and y is in a n-1 dimensional hypercube.

Inverse ptt is basically just computing sums for all subtrees. We do this
by computing cumsum over leaf nodes, then subtracting to get subtree sums.

xs is assumed to have shape [batch_dim, n]
"""
def inverse_ptt(pttargs, x):
    nbatch = x.shape[0]

    x_perm = x[:,pttargs.leaf_permutation]

    x_perm_cumsum = jnp.cumsum(x_perm, dtype=jnp.float64, axis=-1)
    x_perm_cumsum -= x_perm
    cs = jnp.hstack([x_perm_cumsum, jnp.ones((nbatch,1))])

    cs_subtree_max = cs[:,pttargs.max_leaf]
    cs_subtree_min = cs[:,pttargs.min_leaf]
    cs_left_subtree_min = cs[:,pttargs.left_min_leaf]

    u = jnp.float32(cs_subtree_max - cs_subtree_min)
    u_left = jnp.float32(cs_subtree_max - cs_left_subtree_min)
    y = u_left / u

    print((jnp.min(y), jnp.max(y)))

    ladj = -jnp.sum(jnp.log(u), axis=-1)

    return y, ladj

"""
Inefficient forward mode ptt used for initialization.
"""
def ptt(left_index, right_index, leaf_index, y):
    num_nodes = len(left_index)
    n = (num_nodes+1)//2
    u = np.zeros(num_nodes, dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)
    u[0] = 1.0
    k = 0
    for i in range(num_nodes):
        if leaf_index[i] >= 0:
            x[leaf_index[i]] = u[i]
            continue

        u[left_index[i]] = u[i] * y[k]
        u[right_index[i]] = u[i] * (1.0-y[k])
        k += 1

    return x


def approx_log_likelihood(pttargs, α, β, x):
    y, ladj = inverse_ptt(pttargs, x)
    ll = jnp.sum((α - 1.0) * jnp.log(y) + (β - 1.0) * jnp.log(1.0 - y), axis=-1)
    return ll + ladj


class Encoder(nn.Module):
    hiddendim: int
    latentdim: int

    @nn.compact
    def __call__(self, α, β):
        h1 = nn.tanh(nn.Dense(self.hiddendim, name="encoder_layer_1")(jnp.hstack([α, β])))
        μ = nn.Dense(self.latentdim, name="encoder_μ_layer_1")(h1)
        logσ2 = nn.Dense(self.latentdim, name="encoder_logσ_layer_1")(h1)
        return μ, logσ2


class Decoder(nn.Module):
    hiddendim: int
    n: int
    λ_bias_init: Callable

    def setup(self):
        self.lyr1 = nn.Dense(self.hiddendim)
        self.lyrn = nn.Dense(self.n, bias_init=self.λ_bias_init)

    @nn.compact
    def __call__(self, z):
        h1 = nn.leaky_relu(self.lyr1(z))
        # λ = 1.0 + nn.softplus(self.lyrn(h1))
        λ = 1.0 + jnp.exp(self.lyrn(h1))

        return λ


class VAE(nn.Module):
    n: int
    λ_bias_init: Callable
    hiddendim: int = 50
    latentdim: int = 20

    def setup(self):
        self.encoder = Encoder(hiddendim=self.hiddendim, latentdim=self.latentdim)
        self.decoder = Decoder(λ_bias_init=self.λ_bias_init, hiddendim=self.hiddendim, n=self.n)

    def __call__(self, α, β, key):
        z_key, x_key = jax.random.split(key)
        μ, logσ2 = self.encoder(α, β)
        σ = jnp.exp(0.5 * logσ2)
        z = μ + jax.random.normal(z_key, μ.shape, dtype=jnp.float32) * σ

        λ = self.decoder(z)
        x = jax.random.dirichlet(x_key, λ, dtype=jnp.float32)
        # TODO: can try bypassing this and computing dirichlet mean to see if
        # it makes a difference

        return x, μ, logσ2, λ


def elbo(pttargs, x, μ, logσ2, α, β):
    ll = approx_log_likelihood(pttargs, α, β, x)
    σ2 = jnp.exp(logσ2)
    kl = -0.5 * jnp.sum(logσ2 - σ2 - jnp.square(μ) + 1.0, axis=-1)
    return ll - kl


def model(n, λ_bias_init):
    return VAE(λ_bias_init=lambda key, shape: λ_bias_init, n=n)


# @jax.jit
def train_step(optimizer, pttargs, λ_bias_init, α_batch, β_batch, key):
    n = α_batch.shape[1] + 1
    def loss_fn(params):
        x, μ, logσ2, λ = model(n, λ_bias_init).apply({"params": params}, α_batch, β_batch, key)
        neg_elbo = -jnp.mean(elbo(pttargs, x, μ, logσ2, α_batch, β_batch))

        # Just peepin some stuff to see where this goes wrong
        metrics = [
            (jnp.min(x), jnp.max(x)),
            (jnp.min(μ), jnp.max(μ)),
            (jnp.min(logσ2), jnp.max(logσ2)),
            (jnp.min(λ), jnp.max(λ)) ]

        return neg_elbo, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (batch_elbo, metrics), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, batch_elbo, metrics


def batch_data(α, β, nominal_batchsize):
    nobs = α.shape[0]
    nbatch = int(np.ceil(nobs / nominal_batchsize))
    α_batches = np.array_split(α, nbatch, axis=0)
    β_batches = np.array_split(β, nbatch, axis=0)
    return list(zip(α_batches, β_batches))


def load_likap_data(spec):
    αs = []
    βs = []

    for entry in spec["samples"]:
        with h5py.File(entry["file"]) as input:
            αs.append(input["alpha"][:])
            βs.append(input["beta"][:])

    nobs = len(αs)
    perm = list(range(nobs))
    np.random.shuffle(perm)
    return (np.vstack([αs[i] for i in perm]), np.vstack([βs[i] for i in perm]))


def load_ptt_data(ptt_filename):
    with h5py.File(ptt_filename) as input:
        node_js = input["node_js"][:]
        node_parent_idxs = input["node_parent_idxs"][:]
        transcript_ids = list(input["transcript_ids"])

        num_nodes = len(node_js)
        left_index = np.full(num_nodes, -1)
        right_index = np.full(num_nodes, -1)
        leaf_index = np.zeros(num_nodes, np.int)

        for i in range(1, num_nodes):
            parent_idx = node_parent_idxs[i] - 1
            if right_index[parent_idx] == -1:
                right_index[parent_idx] = i
            else:
                left_index[parent_idx] = i

        for i in range(num_nodes):
            leaf_index[i] = node_js[i]-1

    return left_index, right_index, leaf_index


def main():
    ptt_filename = sys.argv[1]
    spec_filename = sys.argv[2]

    spec = yaml.safe_load(open(spec_filename))

    print("loading transformation...")
    left_index, right_index, leaf_index = load_ptt_data(ptt_filename)

    print("loading approximated likelihood...")
    α, β = load_likap_data(spec)

    print("computing inverse transformation params...")
    pttargs = inverse_ptt_params(left_index, right_index, leaf_index)

    nobs = α.shape[0]
    n = α.shape[-1] + 1

    batch_size = 30
    αβ_batches = batch_data(α, β, batch_size)

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)

    # try to find reasonable initial bias for the decoder
    α_mean = np.mean(α, axis=0)
    β_mean = np.mean(β, axis=0)
    y_mean = α_mean / (α_mean + β_mean)
    x_mean = ptt(left_index, right_index, leaf_index, y_mean)

    # # TODO: do ptt and inverse_ptt agree? No they don't. Pretty sure ptt is
    # # right, so that means, quite plausibly, that inverse_ptt is fucked.
    # y_recon = inverse_ptt(pttargs, np.expand_dims(x_mean, axis=0))
    # print(y_mean[0:10])
    # print(np.array(y_recon[0])[0,0:10])
    # return

    λ_bias_init = np.log(x_mean / np.min(x_mean))
    # print(np.min(λ_bias_init))
    # print(np.max(λ_bias_init))
    λ_bias_init = jax.device_put(λ_bias_init)

    # Checking that likelihood actually is improved with a better initialization
    # α_batch, β_batch = αβ_batches[0]
    # print(approx_log_likelihood(pttargs, α_batch, β_batch, jnp.full((1,n), 1/n)))
    # print(approx_log_likelihood(pttargs, α_batch, β_batch, np.expand_dims(x_mean, axis=0)))
    # return

    # init optimizer and such
    init_α = jnp.ones((batch_size, n-1), jnp.float32)
    init_β = jnp.ones((batch_size, n-1), jnp.float32)
    params = model(n, λ_bias_init).init(key, init_α, init_β, init_key)["params"]
    optimizer = flax.optim.Adam(1e-3).create(params)
    optimizer = jax.device_put(optimizer)

    # train
    nepochs = 20
    for epoch in range(nepochs):
        print(f"epoch: {epoch}")
        for (α_batch, β_batch) in αβ_batches:
            α_batch = jax.device_put(α_batch)
            β_batch = jax.device_put(β_batch)
            key, batch_key = jax.random.split(key)
            optimizer, batch_elbo, metrics = train_step(
                optimizer, pttargs, λ_bias_init, α_batch, β_batch, batch_key)
            # print(metrics)
            print(f"mean -elbo {jnp.float32(batch_elbo)}")

if __name__ == "__main__":
    main()
