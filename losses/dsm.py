import torch
import torch.autograd as autograd


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)


def timewise_score_estimation(scorenet, samples, t):
    """
    in objective, T = [0, 1]
    px, qx, xt: (batch_size, 1)
    t: (batch_size, 1)
    """
    px, qx, xt = samples

    term1 = 2 * scorenet(torch.cat([px, torch.zeros_like(px)], dim=-1))
    term2 = 2 * scorenet(torch.cat([qx, torch.ones_like(qx)], dim=-1))
    
    # need to differentiate score wrt t
    t.requires_grad_(True)
    xt_score = scorenet(torch.cat([xt, t], dim=-1)).mean()  # dim = 1
    xt_score_dt = autograd.grad(xt_score, t, create_graph=True)[0]
    term3 = 2 * xt_score_dt
    term4 = (scorenet(torch.cat([xt, t], dim=-1)))**2

    loss = term1 - term2 + term3 + term4

    # 1-d so we can just take the mean rather than summing
    return loss.mean()
