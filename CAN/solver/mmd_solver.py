import torch
import torch.nn as nn
import os
from utils.utils import to_cuda
from torch import optim
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from math import ceil as ceil
from .base_solver import BaseSolver
from discrepancy.mmd import MMD

class MMDSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, fc_domain_map={}, 
                    resume=None, **kwargs):
        super(MMDSolver, self).__init__(net, dataloader, \
                  bn_domain_map=bn_domain_map, fc_domain_map=fc_domain_map, \
                  resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        num_layers = len(self.net.module.FC) + 1
        self.mmd = MMD(num_layers=num_layers, kernel_num=self.opt.MMD.KERNEL_NUM, 
                kernel_mul=self.opt.MMD.KERNEL_MUL, joint=self.opt.MMD.JOINT)

    def solve(self):
        if self.resume:
            self.iters += 1
            self.loop += 1

        self.compute_iters_per_loop()
        while True:
            if self.loop > self.opt.TRAIN.MAX_LOOP: break
            self.update_network()
            self.loop += 1

        print('Training Done!')

    def compute_iters_per_loop(self):
        max_iters = len(self.train_data[self.source_name]['loader'])

        self.iters_per_loop = max(max_iters, 
                    len(self.train_data[self.target_name]['loader']))

        print('Iterations in one loop: %d' % (self.iters_per_loop))


    def update_network(self):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
                 iter(self.train_data[self.source_name]['loader'])
        self.train_data[self.target_name]['iterator'] = \
                 iter(self.train_data[self.target_name]['loader'])

        while not stop:
            loss = 0
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            source_sample = self.get_samples(self.source_name) 
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']

            target_sample = self.get_samples(self.target_name) 
            target_data = target_sample['Img']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_output = self.net(source_data)

            target_data = to_cuda(target_data)
            self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
            target_output = self.net(target_data)

            # compute the cross-entropy loss
            source_preds = source_output['logits']
            ce_loss = self.CELoss(source_preds, source_gt)

            # compute the mmd loss
            source_feats = [source_output[key] for key in source_output if key in self.opt.MMD.ALIGNMENT_FEAT_KEYS]
            target_feats = [target_output[key] for key in target_output if key in self.opt.MMD.ALIGNMENT_FEAT_KEYS]
            assert(len(source_feats) == len(target_feats)), \
                      "The length of source and target features should be the same."
            mmd_loss = self.opt.MMD.LOSS_WEIGHT * self.mmd.forward(source_feats, target_feats)['mmd']

            loss = ce_loss + mmd_loss
            loss.backward()
         
            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss': ce_loss, 'mmd_loss': mmd_loss}
                self.logging(cur_loss, accu)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(self.iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop, 
                              self.iters, self.opt.EVAL_METRIC, accu))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(self.iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

