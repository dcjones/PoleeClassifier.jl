
from collections import namedtuple
import numpy as np
import h5py
import yaml
import sys
import timeit

# need 64 bit floats to compute cumsum differences to do the inverse ptt
from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

PttArgs = namedtuple("PTTArgs", ["leaf_permutation", "max_leaf", "min_leaf", "left_max_leaf"])

"""
Construct necessary parameters to compute the inverse ptt.
"""
def inverse_ptt_params(left_index, right_index, leaf_index):
    num_nodes = len(left_index)
    n = (num_nodes+1)//2

    # compute leaf permutation
    leaf_permutation = np.zeros(n, np.int32)
    k = 0 # leaf node number
    for i in range(num_nodes):
        if leaf_index[i] >= 0:
            leaf_permutation[k] = leaf_index[i]
            k += 1

    # figure out subtree spans for every node
    min_leaf_index = np.zeros([num_nodes], np.int)
    max_leaf_index = np.zeros([num_nodes], np.int)

    k = 0
    for i in range(num_nodes-1, -1, -1):
        if leaf_index[i] >= 0:
            min_leaf_index[i] = k
            max_leaf_index[i] = k
            k += 1
        else:
            min_leaf_index[i] = min_leaf_index[left_index[i]]
            max_leaf_index[i] = max_leaf_index[right_index[i]]
            assert min_leaf_index[i] < max_leaf_index[i]

    # now just compute the indexes we need to compute values for internal nodes
    max_leaf = np.zeros(n-1, np.int32)
    min_leaf = np.zeros(n-1, np.int32)
    left_max_leaf = np.zeros(n-1, np.int32)

    k = 0 # internal node number
    for i in range(num_nodes):
        if leaf_index[i] >= 0:
            continue
        max_leaf[k] = max_leaf_index[i] + 1
        min_leaf[k] = min_leaf_index[i]
        left_max_leaf[k] = max_leaf_index[left_index[i]] + 1
        k += 1

    return PttArgs(
        jax.device_put(leaf_permutation),
        jax.device_put(max_leaf),
        jax.device_put(min_leaf),
        jax.device_put(left_max_leaf))


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

    cs_subtree_min = cs[:,pttargs.min_leaf]
    u_left = jnp.float32(cs[:,pttargs.left_max_leaf] - cs_subtree_min)
    u = jnp.float32(cs[:,pttargs.max_leaf] - cs_subtree_min)
    # u_left = cs[:,pttargs.left_max_leaf] - cs_subtree_min
    # u = cs[:,pttargs.max_leaf] - cs_subtree_min
    y = u_left / u
    ladj = jnp.sum(jnp.log(u), axis=-1)

    return y, ladj


def approx_log_likelihood(pttargs, α, β, x):
    y, ladj = inverse_ptt(pttargs, x)
    ll = jnp.sum((α - 1.0) * jnp.log(y) + (β - 1.0) * jnp.log(1.0 - y), axis=-1)
    ll += ladj
    return ll


def load_likap_data(spec):
    αs = []
    βs = []

    for entry in spec["samples"]:
        with h5py.File(entry["file"]) as input:
            αs.append(input["alpha"][:])
            βs.append(input["beta"][:])

    return (np.vstack(αs), np.vstack(βs))


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


if __name__ == "__main__":
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

    # make some fake data
    x = np.exp(np.float32(np.random.randn(100, n)))
    x = x / np.sum(x, axis=-1, keepdims=True)
    x = jax.device_put(x)

    inverse_ptt_jit = jax.jit(inverse_ptt)
    approx_log_likelihood_jit = jax.jit(approx_log_likelihood)

    # ll = approx_log_likelihood(pttargs, α, β, x)
    # print(ll)

    def run():
        ll = approx_log_likelihood_jit(pttargs, α, β, x)
        ll.block_until_ready()

    print(timeit.timeit(run, number=10))

    # ys = inverse_ptt(pttargs, xs)
    # print(ys)

