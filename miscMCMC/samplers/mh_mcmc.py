'''Metropolis-Hastings MCMC allowing custom jump proposals.
The pdf is required to be written in JAX for automatic differentiation.'''


import numpy as np
from jax import jit, hessian
import jax.random as jr
import random


def MH_MCMC(num_samples,
            ln_pdf_func,
            x0,
            x_mins,
            x_maxs,
            jump_proposals_and_weights=[],
            Fisher_jump_weight=20,
            DE_jump_weight=20):
    
    # initialize samples
    ndim = x0.shape[0]
    samples = np.zeros((num_samples, ndim))
    ln_pdf_vals = np.zeros(num_samples)
    samples[0] = x0
    ln_pdf_vals[0] = ln_pdf_func(samples[0])

    # initialize Fisher information matrix
    Fisher = -hessian(ln_pdf_func)(samples[0])
    vals, vecs = np.linalg.eigh(Fisher)
    Fisher_jumps = np.array(1. / np.sqrt(np.abs(vals)) * vecs.T)

    # Fisher jump proposals
    def Fisher_proposal(x, Fisher_jumps):
        jump = random.choice(Fisher_jumps)
        jump *= random.gauss()
        return x + jump
    
    # initialize history for differential evolution
    len_history = 100
    history = np.random.uniform(x_mins, x_maxs, (len_history, ndim))
    DE_weight = 2.38 / np.sqrt(2. * ndim)
    replacement_ndxs = np.random.choice(len_history, num_samples)

    # DE jump proposal
    @jit
    def DE_proposal(x, i, history):
        draw_key1, draw_key2, weight_key, epsilon_key = jr.split(jr.key(i), 4)
        draw_ndx1 = jr.choice(draw_key1, len_history)
        draw_ndx2 = jr.choice(draw_key2, len_history)
        jump = history[draw_ndx1] - history[draw_ndx2]
        jump *= jr.normal(weight_key) * DE_weight
        jump += jr.normal(epsilon_key) * (1.e-5)
        return x + jump

    # organize jump proposals and weights
    proposals = []
    jump_weights = []
    for proposal_func, weight in jump_proposals_and_weights:
        proposals.append(proposal_func)
        jump_weights.append(weight)
    proposals.append(Fisher_proposal)
    jump_weights.append(Fisher_jump_weight)
    proposals.append(DE_proposal)
    jump_weights.append(DE_jump_weight)
    
    # select jump type at each iteration
    num_jump_proposals = len(proposals)
    jump_ndxs = np.random.choice(np.arange(num_jump_proposals), num_samples,
                                 p=jump_weights/np.sum(jump_weights))
    
    # track jump acceptance rates
    accept_counts = np.zeros(num_jump_proposals)
    reject_counts = np.zeros(num_jump_proposals)

    # store random uniform numbers to decide acceptance
    rand_uniforms = np.random.uniform(size=num_samples)

    # main MCMC loop
    for i in range(num_samples - 1):

        # update progress and Fisher
        if i % 1000 == 0 and i > 0:
            print(f'{np.round(i / num_samples * 100, 3)}%', end='\r')
            Fisher = -hessian(ln_pdf_func)(samples[np.argmax(ln_pdf_vals[:i])])
            vals, vecs = np.linalg.eigh(Fisher)
            Fisher_jumps = 1. / np.sqrt(np.abs(vals)) * vecs.T

        # jump proposal
        jump_ndx = jump_ndxs[i]
        if jump_ndx == num_jump_proposals - 2:  # Fisher jump
            new_state = Fisher_proposal(samples[i], Fisher_jumps)
        elif jump_ndx == num_jump_proposals - 1:  # DE jump
            new_state = DE_proposal(samples[i], i, history)
        else:
            new_state = proposals[jump_ndx](samples[i], i)

        # acceptance probability
        new_ln_pdf_val = ln_pdf_func(new_state)
        acceptance_prob = np.exp(new_ln_pdf_val - ln_pdf_vals[i])

        # decide to accept or reject proposal
        if rand_uniforms[i] < acceptance_prob:  # accept
            accept_counts[jump_ndx] += 1
            samples[i + 1] = new_state
            ln_pdf_vals[i + 1] = new_ln_pdf_val
        else:  # reject
            reject_counts[jump_ndx] += 1
            samples[i + 1] = samples[i]
            ln_pdf_vals[i + 1] = ln_pdf_vals[i]

        # update history for differential evolution
        if i % 100 == 0:
            history[replacement_ndxs[i]] = samples[i + 1]

    # calculate proposal acceptnace rates
    acceptance_rates = accept_counts / (accept_counts + reject_counts)
    print('jump proposal acceptance rates \n')
    for accept_rate, proposal in zip(acceptance_rates, proposals):
        print(f'{proposal.__name__}: {accept_rate}')

    return samples, ln_pdf_vals
    

