import os
import torch
from dataloader import get_loader
from models.model_main import ModelMain
from options import get_parser_main_model
from torch.optim import Adam, AdamW

# Testing (Only accuracy)

def test_main_model(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size_val, 'val')

    model_main = ModelMain(opts).to(device)
    path_ckpt = os.path.join(opts.model_path)
    parameters_all = [{"params": model_main.img_encoder.parameters()}, {"params": model_main.img_decoder.parameters()},
                            {"params": model_main.modality_fusion.parameters()}, {"params": model_main.transformer_main.parameters()},
                            {"params": model_main.transformer_seqdec.parameters()}]

    optimizer = AdamW(parameters_all, lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)
    
    # Check if checkpoint path is correct and the file exists
    if not os.path.isfile(path_ckpt):
        raise FileNotFoundError(f"Checkpoint file not found at {path_ckpt}")
    
    checkpoint = torch.load(path_ckpt, map_location=device)
    model_main.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['opt'])    
    model_main.load_state_dict(checkpoint['model'])
    optimizer.zero_grad()
    optimizer.step()
    with torch.no_grad():
        model_main.eval()
        loss_val = {'img':{'l1':0.0, 'vggpt':0.0}, 'svg':{'total':0.0, 'cmd':0.0, 'args':0.0, 'aux':0.0}}

        for val_idx, val_data in enumerate(val_loader):
            for key in val_data:
                val_data[key] = val_data[key].to(device)
            
            ret_dict_val, loss_dict_val = model_main(val_data, mode='train')
            
            for loss_cat in ['img', 'svg']:
                for key in loss_val[loss_cat]:
                    loss_val[loss_cat][key] += loss_dict_val[loss_cat][key].item()

        for loss_cat in ['img', 'svg']:
            for key in loss_val[loss_cat]:
                loss_val[loss_cat][key] /= len(val_loader)

        val_msg = (
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