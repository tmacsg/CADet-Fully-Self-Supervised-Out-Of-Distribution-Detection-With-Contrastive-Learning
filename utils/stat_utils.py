import torch
import numpy as np
from tqdm import tqdm
from torch.multiprocessing import Pool

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances."""

    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)
    
def calibrate_sigma(sample_1, sample_2):
    # sigma = sqrt(H/2)
    # 1 / 2 sigma**2 = 1 / 2 (H / 2) = 1 / H, H is median of squared distance
    sample_12 = torch.cat((sample_1, sample_2), 0)
    distances = pdist(sample_12, sample_12)
    sigma = 1 / ((distances ** 2).median())
    return sigma
    

def mmd_unbiased(sample_p: torch.Tensor,
                 sample_q: torch.Tensor,
                 sigma: float):
    """Emprical unbiased maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        sample_p: first sample, distribution P, shape [n_1, d]
        sample_q: second sample, distribution Q, shape [n_2, d
        sigma: kernel parameter
    """
    n_1, n_2 = sample_p.shape[0], sample_q.shape[0]
    a00 = 1. / (n_1 * (n_1 - 1))
    a11 = 1. / (n_2 * (n_2 - 1))
    a01 = - 1. / (n_1 * n_2)
 
    sample_pq = torch.cat((sample_p, sample_q), 0)
    distances = pdist(sample_pq, sample_pq)
    kernels =  torch.exp(- sigma * distances ** 2)
    k_1 = kernels[:n_1, :n_1]
    k_2 = kernels[n_1:, n_1:]
    k_12 = kernels[:n_1, n_1:]

    mmd = (2 * a01 * k_12.sum() +
            a00 * (k_1.sum() - torch.trace(k_1)) +
            a11 * (k_2.sum() - torch.trace(k_2)))
    return mmd.item()


def mmd_permutation_test(sample_p: torch.Tensor,
                     sample_q: torch.Tensor,
                     n_permutations: int=500
                    ):   
    sigma = calibrate_sigma(sample_p, sample_q)
    est = mmd_unbiased(sample_p, sample_q, sigma)
    mmd_vals = torch.zeros(n_permutations)
    for i in range(n_permutations):
        sample_pq = torch.cat([sample_p, sample_q], dim=0)
        shuffled_sample_pq = sample_pq[torch.randperm(len(sample_pq))]
        sample_p_new, sample_q_new = torch.split(shuffled_sample_pq, [len(sample_p), len(sample_q)])
        mmd_vals[i] = mmd_unbiased(sample_p_new, sample_q_new, sigma)

    p_value = (torch.sum(mmd_vals >= est).item() + 1) / (n_permutations + 1)
    return p_value, mmd_vals, est

def mmd_permutation_test_cc(sample_p1: torch.Tensor,
                            sample_p2: torch.Tensor,
                            sample_q: torch.Tensor,
                            n_permutations: int=500
                           ):
    sigma1 = calibrate_sigma(sample_p1, sample_q)
    est = mmd_unbiased(sample_p1, sample_q, sigma1)
    sigma2 = calibrate_sigma(sample_p1, sample_p2)
    mmd_vals = torch.zeros(n_permutations)
    for i in range(n_permutations):
        sample_pq = torch.cat([sample_p1, sample_p2], dim=0)
        shuffled_sample_pq = sample_pq[torch.randperm(len(sample_pq))]
        sample_p_new, sample_q_new = torch.split(shuffled_sample_pq, [len(sample_p1), len(sample_p2)])
        mmd_vals[i] = mmd_unbiased(sample_p_new, sample_q_new, sigma2)

    p_value = (torch.sum(mmd_vals >= est).item() + 1) / (n_permutations + 1)
    return p_value, mmd_vals, est
