import math


class LRScheduler:
    def __init__(self, optim, schedule):
        self.optim = optim
        self.schedule = schedule
        self.wp_iter = schedule['warmup_iter']
        self.max_iter = schedule['max_iter']

    def step(self, cur_iter):
        # linear warmup
        if cur_iter < self.wp_iter:
            lrs = self.schedule['start_factor']
            lr_multiplier = lrs + (1 - lrs) * cur_iter / self.wp_iter
        # anneal lr according to scheudle
        else:
            lre = self.schedule['end_factor']
            past = (cur_iter - self.wp_iter) / (self.max_iter - self.wp_iter - 1)
            if self.schedule['type'] == 'cos':
                lr_multiplier = lre + (1 - lre) * (0.5 + 0.5 * math.cos(math.pi * past))
            elif self.schedule['type'] == 'lin':
                thres = 0.15
                lr_multiplier = 1 if past < thres else lre + (1 - lre) * (1 - past) / (1 - thres)
            elif self.schedule['type'] == 'lin0':
                thres = 0.05
                lr_multiplier = 1 if past < thres else lre + (1 - lre) * (1 - past) / (1 - thres)
            elif self.schedule['type'] == 'lin00':
                lr_multiplier = lre + (1 - lre) * (1 - past)
            elif self.schedule['type'].startswith('lin'):
                thres = float(self.schedule['type'][3:])
                lre_mid = lre + (1 - lre) * (1 - thres)
                lre_mid = (1 + lre_mid) / 2
                if past < thres:
                    lr_multiplier = 1 + (lre_mid - 1) * past / thres
                else:
                    lr_multiplier = lre + (lre_mid - lre) * (1 - past) / (1 - thres)
            elif self.schedule['type'] == 'exp':
                thres = 0.15
                lr_multiplier = 1 if past < thres else math.exp((past - thres) / (1 - thres) * math.log(lre))
            else:
                raise NotImplementedError("unknown sche_type {}".format(self.schedule['type']))

        lr_stats = []
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr_multiplier * param_group['init_lr']
            lr_stats.append(param_group['lr'])
        return lr_stats


# def lr_wd_annealing(optimizer, schedule, cur_iter, wp_iter, max_iter):
#     if cur_iter < wp_iter:
#         lr_multiplier = schedule['lr_start_ratio'] + (1-schedule['lr_start_ratio']) * cur_iter/wp_iter
#     else:
#         lre = schedule['lr_end_ratio']
#         past = (cur_iter-wp_iter) / (max_iter-wp_iter-1)
#         if schedule['type'] == 'cos':
#             lr_multiplier = lre + (1-lre) * (0.5 + 0.5 * math.cos(math.pi * past))
#         elif schedule['type'] == 'lin':
#             thres = 0.15
#             lr_multiplier = 1 if past < thres else lre + (1-lre) * (1-past) / (1-thres)
#         elif schedule['type'] == 'lin0':
#             thres = 0.05
#             lr_multiplier = 1 if past < thres else lre + (1-lre) * (1-past) / (1-thres)
#         elif schedule['type'] == 'lin00':
#             lr_multiplier = lre + (1-lre) * (1-past)
#         elif schedule['type'].startswith('lin'):
#             thres = float(schedule['type'][3:])
#             lre_mid = lre + (1-lre) * (1-thres)
#             lre_mid = (1+lre_mid) / 2
#             if past < thres:
#                 lr_multiplier = 1 + (lre_mid-1) * past / thres
#             else:
#                 lr_multiplier = lre + (lre_mid-lre) * (1-past) / (1-thres)
#         elif schedule['type'] == 'exp':
#             thres = 0.15
#             lr_multiplier = 1 if past < thres else math.exp((past-thres) / (1-thres) * math.log(lre))
#         else:
#             raise NotImplementedError("unknown sche_type {}".format(schedule['type']))
#
#     past = cur_iter / (max_iter-1)
#     cur_wd = schedule['wd_end'] + (schedule['wd_start']-schedule['wd_end']) * (0.5 + 0.5 * math.cos(math.pi * past))
#
#     lr_stats = []
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr_multiplier * param_group['init_lr']
#         param_group['weight_decay'] = cur_wd
#         lr_stats.append({
#             'lr': param_group['lr'],
#             'weight_decay': param_group['weight_decay']
#         })
#     return lr_stats
