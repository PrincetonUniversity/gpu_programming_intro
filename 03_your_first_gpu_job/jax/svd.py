from time import perf_counter
import jax
import jax.numpy as jnp

print("JAX version: ", jax.__version__)

N = 4000
key = jax.random.PRNGKey(42)
A = jax.random.normal(key, (N, N))
t0 = perf_counter()
U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
elapsed_time = perf_counter() - t0

print("Execution time: ", elapsed_time)
print("Result: ", jnp.sum(S))
