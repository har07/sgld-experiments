import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
import numpy as np

# Borrowed from https://github.com/henripal/sgld/blob/master/sgld/sgld/sgld_optimizer.py
# with modification to support burn in steps
class SGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement SGLD
    """

    def __init__(self, params, lr=required, addnoise=True, num_burn_in_steps=3000):
        defaults = dict(lr=lr, addnoise=addnoise, num_burn_in_steps=num_burn_in_steps)
        super(SGLD, self).__init__(params, defaults)

    def step(self, lr=None, add_noise = False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["iteration"] = 0
                state["iteration"] += 1

                d_p = p.grad.data
                if group['addnoise'] and state["iteration"] > group["num_burn_in_steps"]:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) / np.sqrt(group['lr'])
                    )

                    noise = langevin_noise.sample().cuda()
                    p.data.add_(d_p + noise, alpha=-group['lr'])
                else:
                    p.data.add_(d_p, alpha=-group['lr'])

        return loss