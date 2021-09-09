import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required

class SGLD(Optimizer):
    """
    Noise calculation adapted from Gustafsson's implementation
    https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/SGLD-64/train.py
    """

    def __init__(self, params, lr=required, train_size=required, weight_decay=0, lr_decay=0, num_burn_in_steps=300):
        defaults = dict(lr=lr, train_size=train_size, weight_decay=weight_decay, lr_decay=lr_decay, 
                        num_burn_in_steps=num_burn_in_steps)
        super(SGLD, self).__init__(params, defaults)

    def step(self, lr=None):
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

                iteration = state["iteration"]
                d_p = p.grad.data
                if iteration <= group["num_burn_in_steps"]:
                    # SGD without noise
                    p.data.add_(d_p, alpha=-group['lr'])
                    return loss

                # if passed burn in steps, do Langevin-like SGD
                weight_decay = group["weight_decay"]
                if weight_decay > 0:
                    d_p.add_(p, alpha=weight_decay)

                lr_decay = group["lr_decay"]
                if lr_decay > 0:
                    lr = lr * (iteration ** -lr_decay)

                size = d_p.size()
                langevin_noise = Normal(
                    torch.zeros(size),
                    torch.ones(size)
                )
                d_p = torch.tensor(lr)*d_p + (1.0/torch.sqrt(group["train_size"])) * torch.sqrt(2.0/torch.tensor(lr)) * \
                        langevin_noise.sample().cuda()
                p.data.add_(-d_p)

        return loss