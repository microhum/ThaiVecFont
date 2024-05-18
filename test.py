import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from dataloader import get_loader
from models.model_main import ModelMain
from models.transformers import denumericalize
from options import get_parser_main_model
from data_utils.svg_utils import render
from models.util_funcs import svg2img, cal_iou

# Testing (Only accuracy)

def test_main_model(opts):
    test_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size, 'test')

    model_main = ModelMain(opts)
    path_ckpt = os.path.join(f"{opts.exp_path}", 'experiments', opts.name_exp, 'checkpoints', opts.name_ckpt)
    model_main.load_state_dict(torch.load(path_ckpt)['model'])
    model_main.cuda()
    model_main.eval() # Testing mode

    with torch.no_grad():
        loss_val = {'img':{'l1':0.0, 'vggpt':0.0}, 'svg':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0},
                                'svg_para':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0}}
        
        for val_idx, val_data in enumerate(test_loader):
            for key in val_data: val_data[key] = val_data[key].cuda()
            ret_dict_val, loss_dict_val = model_main(val_data, mode='val')
            for loss_cat in ['img', 'svg']:
                for key, _ in loss_val[loss_cat].items():
                    loss_val[loss_cat][key] += loss_dict_val[loss_cat][key]

        for loss_cat in ['img', 'svg']:
            for key, _ in loss_val[loss_cat].items():
                loss_val[loss_cat][key] /= len(test_loader) 

        val_msg = (
            f"Val loss img l1: {loss_val['img']['l1']: .6f}, "
            f"Val loss img pt: {loss_val['img']['vggpt']: .6f}, "
            f"Val loss total: {loss_val['svg']['total']: .6f}, "
            f"Val loss cmd: {loss_val['svg']['cmd']: .6f}, "
            f"Val loss args: {loss_val['svg']['args']: .6f}, "
        )

        print(val_msg)
        print(f"l1: {loss_val['img']['l1']: .6f}, pt: {loss_val['img']['vggpt']: .6f}")

def main():
    
    opts = get_parser_main_model().parse_args()
    opts.name_exp = opts.name_exp + '_' + opts.model_name
    experiment_dir = os.path.join(f"{opts.exp_path}","experiments", opts.name_exp)
    print(f"Testing on experiment {opts.name_exp}...")
    # Dump options
    test_main_model(opts)

if __name__ == "__main__":
    main()