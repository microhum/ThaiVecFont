import os
import random
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torchvision.utils import save_image
import wandb
from dataloader import get_loader
from models import util_funcs
from models.model_main import ModelMain
from options import get_parser_main_model
from data_utils.svg_utils import render
from time import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_main_model(opts):
    setup_seed(opts.seed)
    dir_exp = os.path.join(f"{opts.exp_path}", "experiments", opts.name_exp)
    dir_sample = os.path.join(dir_exp, "samples")

    train_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size, opts.mode)
    val_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size_val, 'val')

    model_main = ModelMain(opts)
    model_main.cuda()
    
    parameters_all = [{"params": model_main.img_encoder.parameters()}, {"params": model_main.img_decoder.parameters()},
                            {"params": model_main.modality_fusion.parameters()}, {"params": model_main.transformer_main.parameters()},
                            {"params": model_main.transformer_seqdec.parameters()}]

    optimizer = AdamW(parameters_all, lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)
    
    # For Continue Training
    checkpoint = torch.load(opts.model_path)
    model_main.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['opt'])    

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
    

    for epoch in range(opts.init_epoch, opts.n_epochs):
        t0 = time()
        for idx, data in enumerate(train_loader):
            for key in data: data[key] = data[key].cuda()
            ret_dict, loss_dict = model_main(data)

            loss = opts.loss_w_l1 * loss_dict['img']['l1'] + opts.loss_w_pt_c * loss_dict['img']['vggpt'] + opts.kl_beta * loss_dict['kl'] \
                    + loss_dict['svg']['total'] + loss_dict['svg_para']['total']
            
            # perform optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches_done = epoch * len(train_loader) + idx + 1 
            message = (
                f"Time: {'{} seconds'.format(time() - t0)}, "
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.6f}, "
                f"img_l1_loss: {opts.loss_w_l1 * loss_dict['img']['l1'].item():.6f}, "
                f"img_pt_c_loss: {opts.loss_w_pt_c * loss_dict['img']['vggpt']:.6f}, "
                f"svg_total_loss: {loss_dict['svg']['total'].item():.6f}, "
                f"svg_cmd_loss: {opts.loss_w_cmd * loss_dict['svg']['cmd'].item():.6f}, "
                f"svg_args_loss: {opts.loss_w_args * loss_dict['svg']['args'].item():.6f}, "
                f"svg_smooth_loss: {opts.loss_w_smt * loss_dict['svg']['smt'].item():.6f}, "
                f"svg_aux_loss: {opts.loss_w_aux * loss_dict['svg']['aux'].item():.6f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                f"Step: {batches_done}"
            )
            if batches_done % opts.freq_log == 0:
                print(message)

                
            if opts.freq_val > 0 and batches_done % opts.freq_val == 0:

                with torch.no_grad():
                    model_main.eval()
                    loss_val = {'img':{'l1':0.0, 'vggpt':0.0}, 'svg':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0},
                                'svg_para':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0}}
                    
                    for val_idx, val_data in enumerate(val_loader):
                        for key in val_data: val_data[key] = val_data[key].cuda()
                        ret_dict_val, loss_dict_val = model_main(val_data, mode='val')
                        for loss_cat in ['img', 'svg']:
                            for key, _ in loss_val[loss_cat].items():
                                loss_val[loss_cat][key] += loss_dict_val[loss_cat][key]

                    for loss_cat in ['img', 'svg']:
                        for key, _ in loss_val[loss_cat].items():
                            loss_val[loss_cat][key] /= len(val_loader) 

                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                        f"Val loss img l1: {loss_val['img']['l1']: .6f}, "
                        f"Val loss img pt: {loss_val['img']['vggpt']: .6f}, "
                        f"Val loss total: {loss_val['svg']['total']: .6f}, "
                        f"Val loss cmd: {loss_val['svg']['cmd']: .6f}, "
                        f"Val loss args: {loss_val['svg']['args']: .6f}, "
                    )

                    print(val_msg)



def main():
    opts = get_parser_main_model().parse_args()
    opts.name_exp = opts.name_exp + '_' + opts.model_name
    experiment_dir = os.path.join(opts.exp_path, "experiments", opts.name_exp)
    print(f"Testing on experiment {opts.name_exp}...")

    test_main_model(opts)

if __name__ == "__main__":
    main()