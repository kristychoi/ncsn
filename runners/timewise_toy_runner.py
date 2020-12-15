import logging
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal
from losses.sliced_sm import *
from losses.dsm import *
from models.gmm import GMM, Gaussian, GMMDist, Square, GMMDistAnneal, PeakedGaussians
import matplotlib.pyplot as plt
import torch
import seaborn as sns
sns.set()
sns.set_style('white')

__all__ = ['TimewiseToyRunner']


class ExpActivation(nn.Module):
    def __init__(self, weight = 1):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return torch.exp(self.weight * x)


class TimewiseToyRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    @staticmethod
    def visualize(teacher, model, left_bound=-2., right_bound=2., savefig=None, step=None, device=None):
        grid_size = 100
        mesh = np.linspace(left_bound, right_bound, grid_size)
        mesh = torch.from_numpy(mesh).float()
        if device is not None:
            mesh = mesh.to(device)

        logr_true = teacher.log_density_ratios(mesh)
        print('computing log ratios...WARNING: this is very slow!')
        est_logr = torch.Tensor([numerical_integration(model, x) for x in mesh]).numpy()

        plt.grid(False)
        # plt.axis('off')
        plt.scatter(mesh.cpu().numpy(), logr_true.cpu().numpy(), label='true')
        plt.scatter(mesh.cpu().numpy(), est_logr, label='estimated')
        plt.xlabel('x')
        plt.ylabel('log r(x) = log q(x)/p(x)')
        plt.legend()

        plt.title('Log density ratios', fontsize=16)

        if savefig is not None:
            plt.savefig(savefig + "/{}_data_ratios.png".format(step), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        # i think you can reuse the same mesh: plot the estimated ratios
        plt.grid(False)
        # plt.axis('off')
        plt.scatter(mesh.cpu().numpy(), est_logr)
        plt.xlabel('x')
        plt.ylabel('log r(x) = log q(x)/p(x)')
        plt.legend()

        plt.title('Estimated log density ratios', fontsize=16)

        if savefig is not None:
            plt.savefig(savefig + "/{}_est_data_ratios.png".format(step), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        import os
        np.savez(os.path.join(savefig, 'ratios'), **{'true': logr_true, 'est': est_logr, 'grid': mesh.cpu().numpy()})
        # scores = model(mesh.detach())
        # mesh = mesh.detach().numpy()
        # scores = scores.detach().numpy()
        # TODO: probably want to save scores at some point

    def train(self):
        # this is the score network: added a dim for time component
        n_steps = 10000
        hidden_units = 128
        score = nn.Sequential(
            nn.Linear(1+1, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, hidden_units),
            nn.Softplus(),
            nn.Linear(hidden_units, 1),
            ExpActivation()  # we need the values to "explode"
        )

        # TODO: modify teacher!
        teacher = PeakedGaussians(dim=1)
        optimizer = optim.Adam(score.parameters(), lr=0.001)

        for step in range(n_steps):
            time_index = torch.rand((128, 1))  # 128 time indices? or just 1?
            samples = teacher.sample(128, time_index)

            loss = timewise_score_estimation(score, samples, time_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info('step: {}, loss: {}'.format(step, loss.item()))

            if step % 1000 == 0:
                self.visualize(teacher, score, -8, 8, savefig='/atlas/u/kechoi/ncsn/figures/', step=step)

# can you vectorize this?
def ratio_estimate(t, x, score):
    # HACK
    # NOTE: x here is just one data point! this may break for batches
    t = torch.ones((1,1)) * t
    rx = score(torch.cat([x.view(1,1), t], dim=-1)).item()
    return rx


def numerical_integration(score, x):
    import scipy.integrate as integrate

    result = integrate.quad(ratio_estimate, 0, 1, args=(x, score))[0]
    return result
