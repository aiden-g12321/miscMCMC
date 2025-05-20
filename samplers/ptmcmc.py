'''Parallel tempering MCMC using Fisher and differential evolution jumps.'''


from jax import jit, vmap, hessian
from jax.lax import scan, cond, dynamic_index_in_dim, switch
import jax.numpy as jnp
import jax.random as jr


# parallel tempering chain swap
class PT_swap:

    def __init__(self, num_chains, temperature_ladder, logpdf):
        self.num_chains = num_chains  # number of chains
        self.temperature_ladder = temperature_ladder  # temperature ladder
        self.logpdf = logpdf  # probability density to sample
        self.vectorized_logpdf = jit(vmap(self.logpdf, in_axes=(0, 0)))
        self.chain_ndxs = jnp.arange(self.num_chains)

        self.fast_PT_swap = jit(self.PT_swap)
        

    def PT_swap(self,
                current_states,
                current_logpdfs,
                jump_accept_counts,
                jump_reject_counts,
                keys):

        # scale logpdfs by temperature
        current_logpdfs *= self.temperature_ladder

        def swap_step(carry, j):
            swap_map, jump_accept_counts, jump_reject_counts = carry
            swap_chain = self.num_chains - 2 - j  # reversed index
            i  = dynamic_index_in_dim(swap_map, swap_chain, axis=0, keepdims=False)
            i1 = dynamic_index_in_dim(swap_map, swap_chain + 1, axis=0, keepdims=False)

            # Compute log acceptance ratio
            log_acc_ratio = -current_logpdfs[i] / self.temperature_ladder[swap_chain]
            log_acc_ratio += -current_logpdfs[i1] / self.temperature_ladder[swap_chain + 1]
            log_acc_ratio += current_logpdfs[i1] / self.temperature_ladder[swap_chain]
            log_acc_ratio += current_logpdfs[i] / self.temperature_ladder[swap_chain + 1]
            acc_ratio = jnp.exp(log_acc_ratio)

            key = keys[j]
            rand_val = jr.uniform(key)

            def accept_fn():
                new_swap_map = swap_map.at[swap_chain].set(i1)
                new_swap_map = new_swap_map.at[swap_chain + 1].set(i)
                new_accept = jump_accept_counts.at[-1, swap_chain].add(1)
                return (new_swap_map, new_accept, jump_reject_counts)

            def reject_fn():
                new_reject = jump_reject_counts.at[-1, swap_chain].add(1)
                return (swap_map, jump_accept_counts, new_reject)

            return cond(rand_val <= acc_ratio, accept_fn, reject_fn), None

        # Initialize swap_map
        swap_map = jnp.copy(self.chain_ndxs)

        ((final_swap_map, final_accepts, final_rejects), _) = scan(
            swap_step,
            (swap_map, jump_accept_counts, jump_reject_counts),
            jnp.arange(self.num_chains - 1)
        )

        # Update states based on final swap map
        final_states = current_states[final_swap_map]
        final_logpdfs = self.vectorized_logpdf(final_states, self.temperature_ladder)

        return final_states, final_logpdfs, final_accepts, final_rejects


def mcmc_scan_loop(num_samples,
                   num_chains,
                   logpdf_func,
                   x0,
                   x_mins,
                   x_maxs,
                   jump_proposals=[],
                   Fisher_jump_weight=20,
                   DE_jump_weight=20,
                   PT_swap_weight=20,
                   seed=0):

    # vectorize pdf
    vectorized_logpdf = jit(vmap(logpdf_func, in_axes=(0, 0)))

    # define temperature ladder
    chain_ndxs = jnp.arange(num_chains)
    temperature_ladder = 1.3 ** chain_ndxs
    sqrt_temperatures = jnp.sqrt(temperature_ladder)[:, None]

    # initialize states and logpdfs
    init_states = jnp.tile(x0, (num_chains, 1))
    init_logpdfs = vmap(logpdf_func)(init_states) / temperature_ladder

    # select jump type for every MCMC iteration
    jump_weights = jnp.array([Fisher_jump_weight, DE_jump_weight, PT_swap_weight])
    jump_ndxs = jr.choice(jr.key(seed), jump_weights.shape[0], (num_samples,), p=jump_weights/jnp.sum(jump_weights))

    # initialize jump accept / reject counts
    accept_counts = jnp.zeros((jump_weights.shape[0], num_chains), dtype=jnp.int32)
    reject_counts = jnp.zeros((jump_weights.shape[0], num_chains), dtype=jnp.int32)

    # jump along eigenvectors of Fisher
    Fisher = -hessian(logpdf_func)(x0)
    vals, vecs = jnp.linalg.eigh(Fisher)
    Fisher_jumps = 1. / jnp.sqrt(vals) * vecs.T
    def Fisher_jump(key):
        weight_key, direction_key = jr.split(key, 2)
        jump = jr.choice(direction_key, Fisher_jumps)
        jump *= jr.normal(weight_key)
        return jump
    vectorized_Fisher_jump = jit(vmap(Fisher_jump, in_axes=(0)))
    def Fisher_step(states, logpdfs, iteration, accept_counts, reject_counts, keys):
        # move to new point in parameter space
        jumps = vectorized_Fisher_jump(keys) * sqrt_temperatures
        proposed_states = states + jumps
        # evaluate posterior and acceptance probabilities at new point
        proposed_logpdfs = vectorized_logpdf(proposed_states, temperature_ladder)
        acceptance_probs = jnp.exp(proposed_logpdfs - logpdfs)
        # accept jump
        accept = jr.uniform(jr.key(iteration), num_chains) < acceptance_probs
        new_states = jnp.where(accept[:, None], proposed_states, states)
        new_logpdfs = jnp.where(accept, proposed_logpdfs, logpdfs)
        # update accept and reject counts
        new_accept_counts = accept_counts.at[0].add(accept)
        new_reject_counts = reject_counts.at[0].add(1 - accept)
        return new_states, new_logpdfs, new_accept_counts, new_reject_counts
    fast_Fisher_step = jit(Fisher_step)

    # jump with differential evolution
    len_history = 100
    DE_weight = 2.38 / jnp.sqrt(2. * x0.shape[0])
    history = jr.multivariate_normal(jr.key(seed + 1), x0, jnp.linalg.inv(Fisher), (len_history,))
    def DE_jump(key):
        draw_key1, draw_key2, weight_key, epsilon_key = jr.split(key, 4)
        jump = jr.choice(draw_key1, history) - jr.choice(draw_key2, history)
        jump *= jr.normal(weight_key) * DE_weight
        jump += jr.normal(epsilon_key) * 1.e-4
        return jump
    vectorized_DE_jump = jit(vmap(DE_jump, in_axes=(0)))
    def DE_step(states, logpdfs, iteration, accept_counts, reject_counts, keys):
        # move to new point in parameter space
        jumps = vectorized_DE_jump(keys)
        proposed_states = states + jumps
        # evaluate posterior and acceptance probabilities at new point
        proposed_logpdfs = vectorized_logpdf(proposed_states, temperature_ladder)
        acceptance_probs = jnp.exp(proposed_logpdfs - logpdfs)
        # accept jump
        accept = jr.uniform(jr.key(iteration), num_chains) < acceptance_probs
        new_states = jnp.where(accept[:, None], proposed_states, states)
        new_logpdfs = jnp.where(accept, proposed_logpdfs, logpdfs)
        # update accept and reject counts
        new_accept_counts = accept_counts.at[0].add(accept)
        new_reject_counts = reject_counts.at[0].add(1 - accept)
        return new_states, new_logpdfs, new_accept_counts, new_reject_counts
    fast_DE_step = jit(DE_step)

    # parallel tempering swap
    PT_object = PT_swap(num_chains, temperature_ladder, logpdf_func)

    # one iteration of MCMC
    def mcmc_step(carry, inp):
        states, logpdfs, accept_counts, reject_counts = carry
        iteration, jump_ndx, key = inp
        keys = jr.split(key, num_chains)

        def Fisher_iteration():
            return fast_Fisher_step(states, logpdfs, iteration, accept_counts, reject_counts, keys)

        def DE_iteration():
            return fast_DE_step(states, logpdfs, iteration, accept_counts, reject_counts, keys)

        def PT_iteration():
            return PT_object.fast_PT_swap(states, logpdfs, accept_counts, reject_counts, keys)

        def do_jump():
            return switch(jump_ndx, [Fisher_iteration, DE_iteration, PT_iteration])

        new_states, new_logpdfs, new_accepts, new_rejects = do_jump()

        return (new_states, new_logpdfs, new_accepts, new_rejects), (new_states, new_logpdfs)


    init_carry = init_states, init_logpdfs, accept_counts, reject_counts
    scan_inputs = (jnp.arange(1, num_samples + 1), jump_ndxs, jr.split(jr.key(seed + 2), num_samples))
    (final_states, final_logpdfs, final_accepts, final_rejects), (states, logpdfs) = scan(mcmc_step,
                                                                     init_carry,
                                                                     scan_inputs)

    return states, logpdfs, temperature_ladder



