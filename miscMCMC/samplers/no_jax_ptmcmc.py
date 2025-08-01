'''parallel-tempering MCMC without JAX
'''

import numpy as np


def PTMCMC(num_samples,
           num_chains,
           logpdf_func,
           x0,
           ndim,
           param_mins,
           param_maxs,
           temperature_ladder=None,
           Fisher_weight=20,
           DE_weight=20,
           PT_swap_weight=20):
    
    # define temperature ladder
    if temperature_ladder is None:
        c = 1.5
        chain_ndx = np.arange(num_chains)
        temperature_ladder = c ** chain_ndx
    
    # initialize chains
    samples = np.zeros((num_chains, num_samples, ndim))
    lnlikes = np.zeros((num_chains, num_samples))
    samples[:, 0] = np.random.uniform(param_mins, param_maxs, (num_chains, ndim))
    samples[0, 0] = x0
    lnlikes[:, 0] = np.array([logpdf_func(sample, temperature=temperature)
                              for sample, temperature in zip(samples[:, 0], temperature_ladder)])
    
    # count jump accepts / rejects
    accept_counts = np.zeros((3, num_chains))
    reject_counts = np.zeros((3, num_chains))
    
    # initialize differential evolution objects
    len_history = 1_000
    history = np.random.uniform(param_mins, param_maxs, (len_history, ndim))
    choices1 = np.random.choice(len_history, (num_chains, num_samples))
    choices2 = np.random.choice(len_history, (num_chains, num_samples))
    DE_jump_weight = 2.38 / np.sqrt(2 * ndim)
    weights = np.random.normal(loc=0., scale=1., size=(num_chains, num_samples))

    # compute Fisher using finite differencing
    epsilon = 1.e-3
    Fisher_jumps = np.zeros((num_chains, ndim, ndim))
    def get_Fisher(logpdf_func, params):
        ndim = params.shape[0]
        hessian = np.zeros((ndim, ndim))
        for i in range(ndim):
            dx = np.zeros((ndim))
            dx[i] = epsilon
            for j in range(ndim):
                dy = np.zeros((ndim))
                dy[j] = epsilon
                fpp = logpdf_func(params + dx + dy, temperature=1.0)
                fpm = logpdf_func(params + dx - dy, temperature=1.0)
                fmp = logpdf_func(params - dx + dy, temperature=1.0)
                fmm = logpdf_func(params - dx - dy, temperature=1.0)
                hessian[i, j] = hessian[j, i] = (fpp - fmp - fpm + fmm) / (4. * epsilon**2.)
        return -hessian

    # jump choices
    jump_weights = np.array([Fisher_weight, DE_weight, PT_swap_weight])
    jump_ndxs = np.random.choice(3, num_samples, p=jump_weights/np.sum(jump_weights))
    
    # main MCMC loop
    for i in range(num_samples - 1):
        
        # update progress
        if i % (num_samples // 1000) == 0:
            print(f'{np.round(i / num_samples * 100, 3)}%', end='\r')
            
        # update Fisher matrices occasionally
        if i % 1000 == 0 and Fisher_weight > 0:
            for j in range(num_chains):
                Fisher = get_Fisher(logpdf_func, samples[j, i])
                vals, vecs = np.linalg.eigh(Fisher)
                Fisher_jumps[j] = 1. / np.sqrt(np.abs(vals)) * vecs.T
        
        # Fisher jump
        if jump_ndxs[i] == 0:
            jump_directions = np.random.choice(ndim, num_chains)
            jumps = np.array([Fisher_options[direction] for Fisher_options, direction in zip(Fisher_jumps, jump_directions)])
            jumps = jumps * weights[:, i, None]
            proposals = jumps + samples[:, i]
            lnlike_proposals = np.array([logpdf_func(proposal, temperature=temperature)
                                         for proposal, temperature in zip(proposals, temperature_ladder)])
            acc_probs = np.exp(lnlike_proposals - lnlikes[:, i])
            accepts = np.random.uniform(size=num_chains) < acc_probs
            samples[:, i + 1] = np.copy(np.where(accepts[:, None], proposals, samples[:, i]))
            lnlikes[:, i + 1] = np.copy(np.where(accepts, lnlike_proposals, lnlikes[:, i]))
            accept_counts[0] += accepts
            reject_counts[0] += 1 - accepts

        # differential evolution jump
        if  jump_ndxs[i] == 1:
            first_draws = np.array([history[ndx] for ndx in choices1[:, i]])
            second_draws = np.array([history[ndx] for ndx in choices2[:, i]])
            jumps = DE_jump_weight * (first_draws - second_draws)
            jumps = jumps * weights[:, i, None]
            proposals = jumps + samples[:, i]
            lnlike_proposals = np.array([logpdf_func(proposal, temperature=temperature)
                                         for proposal, temperature in zip(proposals, temperature_ladder)])
            acc_probs = np.exp(lnlike_proposals - lnlikes[:, i])
            accepts = np.random.uniform(size=num_chains) < acc_probs
            samples[:, i + 1] = np.copy(np.where(accepts[:, None], proposals, samples[:, i]))
            lnlikes[:, i + 1] = np.copy(np.where(accepts, lnlike_proposals, lnlikes[:, i]))
            accept_counts[1] += accepts
            reject_counts[1] += 1 - accepts
            history[np.random.choice(len_history)] = samples[0, i + 1]
        
        # parallel tempering swap
        if jump_ndxs[i] == 2:
            swap_map = np.copy(chain_ndx)
            log_Ls = np.copy(lnlikes[:, i]) * temperature_ladder
            
            for swap_chain in range(num_chains - 2, -1, -1):
                assert swap_map[swap_chain] == swap_chain
                log_acc_ratio = -log_Ls[swap_map[swap_chain]] / temperature_ladder[swap_chain]
                log_acc_ratio += -log_Ls[swap_map[swap_chain + 1]] / temperature_ladder[swap_chain + 1]
                log_acc_ratio += log_Ls[swap_map[swap_chain + 1]] / temperature_ladder[swap_chain]
                log_acc_ratio += log_Ls[swap_map[swap_chain]] / temperature_ladder[swap_chain + 1]

                acc_decide = np.log(np.random.uniform(0.0, 1.0, 1))
                if acc_decide<=log_acc_ratio:
                    swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
                    accept_counts[2, swap_chain] += 1
                else:
                    accept_counts[2, swap_chain] += 1

            Fisher_jumps_new = np.zeros_like(Fisher_jumps)
            for j in range(num_chains):
                samples[j, i + 1] = samples[swap_map[j], i]
                Fisher_jumps_new[j] = Fisher_jumps[swap_map[j]]
                lnlikes[j, i + 1] = lnlikes[swap_map[j], i] * temperature_ladder[swap_map[j]] / temperature_ladder[j]
            Fisher_jumps = np.copy(Fisher_jumps_new)
                    
    # calculate acceptance rates
    jump_names = np.array(['Fisher', 'DE', 'PT'])
    reject_counts[2, -1] += 1
    acceptance_rates = accept_counts / (accept_counts + reject_counts)
    print('jump acceptance rates:')
    for accept_rate, name in zip(acceptance_rates, jump_names):
        print(f'{name} = {np.round(accept_rate, 3)}\n')
    
    return samples, lnlikes, temperature_ladder