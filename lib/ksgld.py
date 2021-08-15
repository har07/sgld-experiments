import torch
import torch.nn.functional as F

from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required


# Borrowed from https://github.com/Thrandis/EKFAC-pytorch/blob/master/kfac.py
# with modification to do actual parameter updates instead of just preconditioning gradient
# and support burn in steps.
class KSGLD(Optimizer):

    def __init__(self, net, eps, lr=required, train_size=required, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False, add_noise=True, num_burn_in_steps=300):
        """ K-FAC Optimizer for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            lr (float): Learning rate.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
            num_burn_in_steps (int): Perform K-FAC update without gaussian noise
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.num_burn_in_steps = num_burn_in_steps
        self.add_noise = add_noise
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        
        defaults = dict(lr=lr, train_size=train_size)
        super(KSGLD, self).__init__(self.params, defaults)

    @torch.no_grad()
    def step(self, lr=None, update_stats=True, update_params=True):
        """Performs one step of optimization."""
        loss = None
        fisher_norm = 0.
        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            else:
                lr = group['lr']
            # Getting parameters
            if len(group['params']) > 2:
                print("len(group['params']) is more than 2!!!: ", len(group['params']))
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            if not 'step' in state:
                state['step'] = 0
            state['step'] += 1
            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)
            if update_params:
                # Preconditionning
                gw, gb, noise_term, noise_bias = self._precond(weight, bias, group, state)
                if noise_term is not None:
                    state['noise_term'] = noise_term
                    # print('noise_term variance: ', torch.var(noise_term))
                    # print('noise_term std: ', torch.std(noise_term))
                if noise_bias is not None and bias is not None:
                    state['noise_bias'] = noise_bias
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                weight.grad = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad = gb
            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    param.grad *= scale
        if update_stats:
            self._iteration_counter += 1
        
        # update network parameters using simple SGD
        # TODO: update network params in the same
        # `for group in self.param_groups:`-loop as updating gradient
        # because noise_term and noise_bias information only available there.
        # Or maybe store noise in group variable?
        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            for i,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                
                if self.add_noise and state["step"] > self.num_burn_in_steps:
                    if i == 0:
                        noise = state['noise_term']
                    else:
                        noise = state['noise_bias']

                    # SGLD update-style:
                    # noise.mul_(torch.tensor(group['lr']).mul(2).sqrt())/group["train_size"]
                    # d_p = d_p.mul(group['lr']) + noise
                    # p.add_(-d_p)

                    # pSGLD update style:
                    # noise = noise.mul(torch.tensor(2*lr).sqrt()).mul(lr).div(group["train_size"])
                    # p.add_(d_p + noise.div(group['lr']), alpha=-group['lr'])

                    # Henripal's KSGLD update style:
                    p.add_(d_p.div(2) + noise.mul(group['lr']).div(group["train_size"]), alpha=-group['lr'])

                    # print('step: ', state['step'])
                    # print('noise_term variance: ', torch.var(noise))
                    # print('noise_term std: ', torch.std(noise))
                else:
                    p.add_(d_p, alpha=-group['lr'])
        
        return loss

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning.
        `noise_term` and `noise_bias` generated from standard normal distribution
        with shape corresponds to gradient of weight and bias, respectively. 
        Noise preconditioned following the exact procedure of the gradient preconditioning
        """
        noise_term = None
        noise_bias = None
        if group['layer_type'] == 'Conv2d' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        noize_size = g.size()
        g = torch.mm(torch.mm(iggt, g), ixxt)
        # add preconditioned noise terms
        # this only implemented for Linear Layer. Conv2d layer with noise 
        # implemented with SUA approximation only.
        if self.add_noise and state["step"] > self.num_burn_in_steps:
            langevin_noise = Normal(
                        torch.zeros(noize_size).cuda(),
                        torch.ones(noize_size).cuda()
                    )
            noise_term = langevin_noise.sample()
            noise_term = torch.mm(torch.mm(iggt, noise_term), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
            if noise_term is not None:
                noise_bias = noise_term[:, -1].view(*bias.shape)
                noise_term = noise_term[:, :-1]
        else:
            gb = None

        g = g.contiguous().view(*s)
        if noise_term is not None:
            noise_term = noise_term.view(*s)

        # if noise_term is not None:
        #     print('noise_term variance: ', torch.var(noise_term))
        #     print('noise_term std: ', torch.std(noise_term))
        return g, gb, noise_term, noise_bias

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA.
        `noise_term` and `noise_bias` generated from standard normal distribution
        with shape corresponds to gradient of weight and bias, respectively. 
        Noise preconditioned following the exact procedure of the gradient preconditioning.
        """
        noise_term = None
        noise_bias = None
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        # if bias not None, merge grad of weight and grad of bias into `g`
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        noise_size = g.size()
        # precondition `g` with `ixxt` and `iggt`
        # `iggt` * `ixxt` * `g`
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state['num_locations']
        # add preconditioned noise terms
        if self.add_noise and state["step"] > self.num_burn_in_steps:
            langevin_noise = Normal(
                        torch.zeros(noise_size).cuda(),
                        torch.ones(noise_size).cuda()
                    )
            # noise_term = langevin_noise.sample().mul(avg.reciprocal().mul(2*lr).sqrt())
            noise_term = langevin_noise.sample()
            noise_term = torch.mm(ixxt, noise_term.contiguous().view(-1, s[0]*s[2]*s[3]))
            noise_term = noise_term.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
            noise_term = torch.mm(iggt, noise_term.view(s[0], -1)).view(s[0], -1, s[2], s[3])
            noise_term /= state['num_locations']
        # if bias not None, split back grad of weight and grad of bias from `g`
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
            if noise_term is not None:
                noise_bias = noise_term[:, -1, s[2]//2, s[3]//2]
                noise_term = noise_term[:, :-1]
        else:
            gb = None

        # if noise_term is not None:
        #     print('noise_term variance (SUA): ', torch.var(noise_term))
        #     print('noise_term std (SUA): ', torch.std(noise_term))
        return g, gb, noise_term, noise_bias

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.t()
            state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt
    
    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
