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
                keys,
                Fisher_jumps,
                history):

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

        return final_states, final_logpdfs, final_accepts, final_rejects, Fisher_jumps, history


# parallel tempering MCMC
def ptmcmc_sampler(num_samples,
                   num_chains,
                   logpdf_func,
                   x0,
                   x_mins,
                   x_maxs,
                   Fisher_jump_weight=20,
                   DE_jump_weight=20,
                   PT_swap_weight=20,
                   seed=0):

    # vectorize pdf
    vectorized_logpdf = jit(vmap(logpdf_func, in_axes=(0, 0)))

    # define temperature ladder
    chain_ndxs = jnp.arange(num_chains)
    temperature_ladder = 1.5 ** chain_ndxs
    sqrt_temperatures = jnp.sqrt(temperature_ladder)[:, None]

    # initialize states and logpdfs
    init_states = jnp.tile(x0, (num_chains, 1))
    init_logpdfs = vmap(logpdf_func)(init_states, temperature_ladder)

    # select jump type for every MCMC iteration
    jump_weights = jnp.array([Fisher_jump_weight, DE_jump_weight, PT_swap_weight])
    jump_ndxs = jr.choice(jr.key(seed), jump_weights.shape[0], (num_samples,),
                          p=jump_weights/jnp.sum(jump_weights))

    # initialize jump accept / reject counts
    accept_counts = jnp.zeros((jump_weights.shape[0], num_chains), dtype=jnp.int32)
    reject_counts = jnp.zeros((jump_weights.shape[0], num_chains), dtype=jnp.int32)

    # jump along eigenvectors of Fisher
    def get_Fisher_jumps(x):
        Fisher = -hessian(logpdf_func)(x)
        vals, vecs = jnp.linalg.eigh(Fisher)
        Fisher_jumps = 1. / jnp.sqrt(jnp.abs(vals)) * vecs.T
        return Fisher_jumps
    fast_get_Fisher_jumps = jit(get_Fisher_jumps)
    init_Fisher_jumps = fast_get_Fisher_jumps(x0)
    def pick_Fisher_jump(Fisher_jumps, key):
        weight_key, direction_key = jr.split(key)
        jump = jr.choice(direction_key, Fisher_jumps)
        jump *= jr.normal(weight_key)
        return jump
    vectorized_pick_Fisher_jump = jit(vmap(pick_Fisher_jump, in_axes=(None, 0)))
    def Fisher_step(states, logpdfs, iteration,
                    accept_counts, reject_counts, keys, Fisher_jumps,
                    history, Fisher_update_rate=0.001):
        # decide whether or not to update Fisher
        update_Fisher = jr.uniform(jr.key(iteration + 2)) < Fisher_update_rate
        def update_Fisher_case():
            return fast_get_Fisher_jumps(states[0])
        def no_update_Fisher_case():
            return Fisher_jumps
        # move to new point in parameter space
        Fisher_jumps = cond(update_Fisher, update_Fisher_case, no_update_Fisher_case)
        # jumps = vectorized_pick_Fisher_jump(Fisher_jumps, keys) * sqrt_temperatures
        jumps = vectorized_pick_Fisher_jump(Fisher_jumps, keys)
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
        return new_states, new_logpdfs, new_accept_counts, new_reject_counts, Fisher_jumps, history
    fast_Fisher_step = jit(Fisher_step)

    # jump with differential evolution
    len_history = 100
    DE_weight = 2.38 / jnp.sqrt(2. * x0.shape[0])
    # init_history = jr.multivariate_normal(key=jr.key(seed + 1),
    #                                       mean=x0,
    #                                       cov=jnp.linalg.inv(-hessian(logpdf_func)(x0)),
    #                                       shape=(len_history,),
    #                                       method='svd')
    init_history = jr.uniform(jr.key(seed + 1), shape=(len_history, x0.shape[0]), minval=x_mins, maxval=x_maxs)
    def DE_jump(key, history):
        draw_key1, draw_key2, weight_key, epsilon_key = jr.split(key, 4)
        jump = jr.choice(draw_key1, history) - jr.choice(draw_key2, history)
        jump *= jr.normal(weight_key) * DE_weight
        jump += jr.normal(epsilon_key) * 1.e-4
        return jump * 10
    vectorized_DE_jump = jit(vmap(DE_jump, in_axes=(0, None)))
    def DE_step(states, logpdfs, iteration, accept_counts, reject_counts, keys, Fisher_jumps, history):
        # move to new point in parameter space
        jumps = vectorized_DE_jump(keys, history)
        proposed_states = states + jumps
        # evaluate posterior and acceptance probabilities at new point
        proposed_logpdfs = vectorized_logpdf(proposed_states, temperature_ladder)
        acceptance_probs = jnp.exp(proposed_logpdfs - logpdfs)
        # accept jump
        accept = jr.uniform(jr.key(iteration), num_chains) < acceptance_probs
        new_states = jnp.where(accept[:, None], proposed_states, states)
        new_logpdfs = jnp.where(accept, proposed_logpdfs, logpdfs)
        # update accept and reject counts
        new_accept_counts = accept_counts.at[1].add(accept)
        new_reject_counts = reject_counts.at[1].add(1 - accept)
        # update history
        history = history.at[jr.choice(jr.key(iteration + 1), len_history)].set(states[0])
        return new_states, new_logpdfs, new_accept_counts, new_reject_counts, Fisher_jumps, history
    fast_DE_step = jit(DE_step)

    # parallel tempering swap
    PT_object = PT_swap(num_chains, temperature_ladder, logpdf_func)

    # one iteration of MCMC
    def mcmc_step(carry, inp):
        states, logpdfs, accept_counts, reject_counts, Fisher_jumps, history = carry
        iteration, jump_ndx, key = inp
        keys = jr.split(key, num_chains)

        def Fisher_iteration():
            return fast_Fisher_step(states, logpdfs, iteration, accept_counts, reject_counts, keys, Fisher_jumps, history)

        def DE_iteration():
            return fast_DE_step(states, logpdfs, iteration, accept_counts, reject_counts, keys, Fisher_jumps, history)

        def PT_iteration():
            return PT_object.fast_PT_swap(states, logpdfs, accept_counts, reject_counts, keys, Fisher_jumps, history)

        def do_jump():
            return switch(jump_ndx, [Fisher_iteration, DE_iteration, PT_iteration])

        new_states, new_logpdfs, new_accepts, new_rejects, Fisher_jumps, history = do_jump()

        return (new_states, new_logpdfs, new_accepts, new_rejects, Fisher_jumps, history), (new_states, new_logpdfs)


    # initial objects to carry through scan
    init_carry = init_states, init_logpdfs, accept_counts, reject_counts, init_Fisher_jumps, init_history
    # inputs to each scan iteration
    scan_inputs = (jnp.arange(1, num_samples + 1), jump_ndxs, jr.split(jr.key(seed + 2), num_samples))
    
    # do MCMC using jax.lax.scan
    ((final_states, final_logpdfs, final_accepts, final_rejects, 
     final_Fisher_jumps, final_history),
    (states, logpdfs)) = scan(mcmc_step, init_carry, scan_inputs)

    # calculate proposal acceptance rates
    jump_names = ['Fisher', 'DE', 'PT swap']
    acceptance_rates = final_accepts / (final_accepts + final_rejects)
    print('jump acceptance rates\n')
    _ = [print(f'{name}: {acc_rates}\n') for name, acc_rates
         in zip(jump_names, jnp.round(acceptance_rates, 3))]

    return states, logpdfs, temperature_ladder



