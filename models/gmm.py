import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import torch.autograd as autograd


class GaussianDist(object):
    def __init__(self, dim, ill_conditioned):
        cov = torch.eye(dim)
        # cov = torch.range(1, dim).diag()
        if ill_conditioned:
            cov[dim // 2:, dim // 2:] = 0.0001 * torch.eye(dim // 2)
        # mean = 0 * torch.ones(dim)
        mean = torch.range(1, dim) / 10
        m = MultivariateNormal(mean, cov)
        self.gmm = m

    def sample(self, n):
        return self.gmm.sample(n)

    def log_pdf(self, x):
        return self.gmm.log_prob(x)

class GMMDistAnneal(object):
    def __init__(self, dim):
        self.mix_probs = torch.tensor([0.8, 0.2])
        # self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        # self.mix_probs = torch.tensor([0.1, 0.1, 0.8])
        # self.means = torch.stack([5 * torch.ones(dim), torch.zeros(dim), -torch.ones(dim) * 5], dim=0)
        self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        self.sigma = 1

    def sample(self, n, sigma=1):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        return torch.randn_like(means) * sigma + means


    def log_prob(self, samples, sigma=1):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * sigma ** 2) - 0.5 * np.log(
                2 * np.pi * sigma ** 2)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp

    def score(self, samples, sigma=1):
        with torch.enable_grad():
            samples = samples.detach()
            samples.requires_grad_(True)
            log_probs = self.log_prob(samples, sigma).sum()
            return autograd.grad(log_probs, samples)[0]


class GMMDist(object):
    def __init__(self, dim):
        self.mix_probs = torch.tensor([0.8, 0.2])
        # self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        # self.mix_probs = torch.tensor([0.1, 0.1, 0.8])
        # self.means = torch.stack([5 * torch.ones(dim), torch.zeros(dim), -torch.ones(dim) * 5], dim=0)
        self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        self.sigma = 1
        self.std = torch.stack([torch.ones(dim) * self.sigma for i in range(len(self.mix_probs))], dim=0)

    def sample(self, n):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log(
                2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp


class PeakedGaussians(object):
    """
    The ratio we are estimating is: r(x) = log q(x) - log p(x)
    where q(x) = N(0, 1e-6) and p(x) = N(0, 1)
    q(x) corresponds to T = 1, p(x) corresponds to T = 0
    """
    def __init__(self, dim):
        self.means = [0, 0]
        # self.sigmas = [1e-6, 1]
        self.sigmas = [1, 1e-6]  # for consistency with tre
        self.dim = dim

    def sample_sequence_on_the_fly(self, px, qx, t):
        # note: t is functioning as \alpha(t) here
        return torch.sqrt(1 - t**2) * px + (t * qx)

    def sample(self, n, t):
        px = self.means[0] + torch.randn((n, self.dim)) * self.sigmas[0]
        qx = self.means[1] + torch.randn((n, self.dim)) * self.sigmas[1]
        xt = self.sample_sequence_on_the_fly(px, qx, t)

        return px, qx, xt

    def log_density_ratios(self, samples):
        log_p = Normal(0,self.sigmas[0]).log_prob(samples)
        log_q = Normal(0,self.sigmas[1]).log_prob(samples)
        # log_p = torch.distributions.Normal(0,1).log_prob(mesh)
        # log_q = torch.distributions.Normal(0,1e-6).log_prob(mesh)
        log_ratios = log_q - log_p

        return log_ratios


class Square(object):
    def __init__(self, range=4.):
        self.range = range

    def sample(self, n):
        n = n[0]
        rands = torch.rand(n, 2)
        samples = (rands - 0.5) * self.range * 2
        return samples

    def log_prob(self, samples):
        range_th = torch.tensor(self.range)
        idx = (samples[:, 0] <= range_th) & (samples[:, 0] >= -range_th) & (samples[:, 1] <= range_th) & (samples[:, 1] >= -range_th)
        results = torch.zeros(samples.shape[0])
        results[~idx] = -1e10
        results[idx] = np.log(1 / (self.range * 2) ** 2)

        return results


class GMM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = torch.randn(3, dim)
        self.mean[0, :] += 1
        self.mean[2, :] -= 1
        self.mean = nn.Parameter(self.mean)
        self.log_std = nn.Parameter(torch.randn(3, dim))
        self.mix_logits = nn.Parameter(torch.randn(3))

    def forward(self, X):
        energy = (X.unsqueeze(1) - self.mean) ** 2 / (2 * (2 * self.log_std).exp()) + np.log(
            2 * np.pi) / 2. + self.log_std
        log_prob = -energy.sum(dim=-1)
        mix_probs = F.log_softmax(self.mix_logits)
        log_prob += mix_probs
        log_prob = torch.logsumexp(log_prob, dim=-1)
        return log_prob


class Gaussian(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))

    def forward(self, X):
        energy = (X - self.mean) ** 2 / (2 * (2 * self.log_std).exp()) + np.log(2 * np.pi) / 2. + self.log_std
        log_prob = -energy
        return log_prob


class Gaussian4SVI(nn.Module):
    def __init__(self, batch_size, dim):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(batch_size, dim))
        self.mean = nn.Parameter(torch.zeros(batch_size, dim))

    def forward(self, X):
        return self.mean, self.log_std
