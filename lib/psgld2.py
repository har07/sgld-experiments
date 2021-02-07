import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
import numpy as np

# Borrowed from https://github.com/henripal/sgld/blob/master/sgld/sgld/sgld_optimizer.py
# with modification to support burn in steps
class pSGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement pSGLD
    The RMSprop preconditioning code is mostly from pytorch rmsprop implementation.
    """

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, centered=False, 
                    addnoise=True, num_burn_in_steps=num_burn_in_steps):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, centered=centered, 
                        addnoise=addnoise, num_burn_in_steps=num_burn_in_steps)
        super(pSGLD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

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
                d_p = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1
                
                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1-alpha, d_p, d_p)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    
                
                if group['addnoise'] and state["iteration"] > group["num_burn_in_steps"]:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size).cuda(),
                        torch.ones(size).cuda().div_(group['lr']).div_(avg).sqrt()
                    )
                    p.data.add_(-group['lr'],
                                d_p.div_(avg) + langevin_noise.sample())
                else:
                    #p.data.add_(-group['lr'], d_p.div_(avg))
                    p.data.addcdiv_(-group['lr'], d_p, avg)

        return loss