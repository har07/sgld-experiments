import numpy as np

def update_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def lr_poly(start_lr, end_lr, step, decay_steps, power):
  step = min(step, decay_steps)
  return ((start_lr - end_lr) *
            (1 - step / decay_steps) ** (power)
           ) + end_lr

def lr_cyclic(M, lr_0, T, epoch, num_batch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0

    return lr

def update_lr(schedule_name, optimizer, accept_lr, initial_lr, current_lr, current_epoch, \
    max_epoch, num_batch, batch_idx, **kwargs):

    should_update_lr = False
    if schedule_name == "cyclic":
        should_update_lr = True
        T = max_epoch * num_batch
        current_lr = lr_cyclic(kwargs["cycle"], initial_lr, T, current_epoch, num_batch, batch_idx)
    elif schedule_name == "welling_teh_2011":
        should_update_lr = True
        current_lr = kwargs['a'] * (kwargs['b'] + current_epoch) ** -kwargs['gamma']
    elif schedule_name == "polynomial_decay":
        should_update_lr = True
        current_lr = lr_poly(initial_lr, kwargs['min_lr'], current_epoch, max_epoch, kwargs['decay'])
    elif schedule_name == "block_decay":
        if ((current_epoch) % kwargs['block_size']) == 0:
            should_update_lr = True
            current_lr = current_lr * kwargs['block_decay']
    else:
        raise ValueError(f"unknown learning rate schedule {schedule_name} !")

    # if optimizer.step doesn't accept learning rate param, 
    # update optimizer's internal 'lr' parameter directly
    if should_update_lr and not accept_lr:
        optimizer = update_optimizer(optimizer, current_lr)

    return current_lr