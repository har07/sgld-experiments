import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required

class pSGLD(Optimizer):
    """
    Translated from official implementation (matlab), with modification to support burn in steps
    https://github.com/ChunyuanLI/pSGLD/blob/master/pSGLD_DNN/algorithms/SGLD_RMSprop.m
    """

    def __init__(self, params, lr=required, train_size=required, rmsprop_decay=.99, eps=1e-1, 
                    weight_decay=0, lr_offset=0, lr_decay=0, num_burn_in_steps=300):
        defaults = dict(lr=lr, train_size=train_size, rmsprop_decay=rmsprop_decay, eps=eps, weight_decay=weight_decay, 
                        lr_offset=lr_offset, lr_decay=lr_decay, num_burn_in_steps=num_burn_in_steps)
        super(pSGLD, self).__init__(params, defaults)

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

                # if passed burn in steps, do Langevin-like SGD with RMSProp preconditioner
                weight_decay = group["weight_decay"]
                if weight_decay > 0:
                    d_p.add_(p, alpha=weight_decay)

                lr_offset = group["lr_offset"]
                lr_decay = group["lr_decay"]
                if lr_offset > 0:
                    lr * ((iteration+lr_offset) ** -lr_decay)
                elif lr_decay > 0:
                    lr = lr * (iteration ** -lr_decay)

                if not "history" in state:
                    # state["history"] = d_p.mul(d_p)
                    state["history"] = torch.zeros_like(p.data)

                rmsd = group["rmsprop_decay"]
                eps = group["eps"]
                # state["history"] = rmsd * state["history"] + (1-rmsd) * (d_p ** 2)
                # precond = (torch.tensor(eps) + torch.sqrt(d_p))
                state["history"] = state["history"].mul(rmsd).addcmul(d_p, d_p, value=1-rmsd)
                precond = d_p.rsqrt().add(eps)

                size = d_p.size()
                langevin_noise = Normal(
                    torch.zeros(size),
                    torch.ones(size)
                )
                # d_p = torch.tensor(lr)*d_p / precond + torch.sqrt(2*torch.tensor(lr)/precond) * \
                #         langevin_noise.sample().cuda()/group["train_size"]
                noise_term = precond.mul(2*lr).sqrt().mul(langevin_noise.sample().cuda().div(group["train_size"]))
                new_grad = d_p.mul(lr).mul(precond).add(noise_term)
                p.data.add_(new_grad, alpha=-1)

        return loss