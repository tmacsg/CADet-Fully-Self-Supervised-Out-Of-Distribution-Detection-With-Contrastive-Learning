import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class Kernel:
    def __call__(self, 
                 sample_p: torch.Tensor, 
                 sample_q: torch.Tensor
                 ) -> torch.Tensor:
        raise NotImplementedError

class GuassianKernel(Kernel):
    def __init__(self):
        super().__init__()
    
    def __call__(self, sample_p, sample_q):
        distances, sigma = self._calibrate_sigma(sample_p, sample_q)
        kernels =  torch.exp(- sigma * distances ** 2)
        return kernels
              
    def _calibrate_sigma(self, sample_p, sample_q):
        sample_pq = torch.cat((sample_p, sample_q), 0)
        distances = self._pdist(sample_pq, sample_pq)
        sigma = 1 / ((distances ** 2).median())
        return distances, sigma
    
    def _pdist(self, sample_p, sample_q, eps=1e-5):
        n_1, n_2 = sample_p.size(0), sample_q.size(0)
        norms_1 = torch.sum(sample_p**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_q**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_p.mm(sample_q.t())
        return torch.sqrt(eps + torch.abs(distances_squared))

class CosineKernel(Kernel):
    def __init__(self):
        super().__init__()
        
    def __call__(self, sample_p, sample_q):
        sample_pq = torch.cat((sample_p, sample_q), 0)
        sample_pq = F.normalize(sample_pq, dim=1)
        kernels = sample_pq @ sample_pq.t()
        return kernels  

def mmd_unbiased(sample_p: torch.Tensor,
                 sample_q: torch.Tensor,
                 kernel: Kernel):
    """Emprical unbiased maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        sample_p: first sample, distribution P, shape [n_1, d]
        sample_q: second sample, distribution Q, shape [n_2, d
        kernel: kernel parameter
    """
    n_1, n_2 = sample_p.shape[0], sample_q.shape[0]
    a00 = 1. / (n_1 * (n_1 - 1))
    a11 = 1. / (n_2 * (n_2 - 1))
    a01 = - 1. / (n_1 * n_2) 
    kernels =  kernel(sample_p, sample_q)
    k_1 = kernels[:n_1, :n_1]
    k_2 = kernels[n_1:, n_1:]
    k_12 = kernels[:n_1, n_1:]

    mmd = (2 * a01 * k_12.sum() +
            a00 * (k_1.sum() - torch.trace(k_1)) +
            a11 * (k_2.sum() - torch.trace(k_2)))
    return mmd.item()


def mmd_permutation_test(sample_p: torch.Tensor,
                     sample_q: torch.Tensor,
                     n_permutations: int=500,
                     kernel: Kernel = CosineKernel()
                    ):   
    est = mmd_unbiased(sample_p, sample_q, kernel)
    mmd_vals = torch.zeros(n_permutations)
    for i in range(n_permutations):
        sample_pq = torch.cat([sample_p, sample_q], dim=0)
        shuffled_sample_pq = sample_pq[torch.randperm(len(sample_pq))]
        sample_p_new, sample_q_new = torch.split(shuffled_sample_pq, [len(sample_p), len(sample_q)])
        mmd_vals[i] = mmd_unbiased(sample_p_new, sample_q_new, kernel)

    p_value = (torch.sum(mmd_vals >= est).item() + 1) / (n_permutations + 1)
    return p_value, mmd_vals, est

def mmd_permutation_test_cc(sample_p1: torch.Tensor,
                            sample_p2: torch.Tensor,
                            sample_q: torch.Tensor,
                            n_permutations: int=500,
                            kernel: Kernel = CosineKernel()
                           ):
    est = mmd_unbiased(sample_p1, sample_q, kernel)
    mmd_vals = torch.zeros(n_permutations)
    for i in range(n_permutations):
        sample_pq = torch.cat([sample_p1, sample_p2], dim=0)
        shuffled_sample_pq = sample_pq[torch.randperm(len(sample_pq))]
        sample_p_new, sample_q_new = torch.split(shuffled_sample_pq, [len(sample_p1), len(sample_p2)])
        mmd_vals[i] = mmd_unbiased(sample_p_new, sample_q_new, kernel)

    p_value = (torch.sum(mmd_vals >= est).item() + 1) / (n_permutations + 1)
    return p_value, mmd_vals, est