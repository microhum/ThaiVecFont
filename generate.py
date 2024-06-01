import fontTools
import os
import shutil
import typing
import PIL
from PIL import Image, ImageDraw, ImageFont
from data_utils.convert_ttf_to_sfd import convert_mp
from data_utils.write_glyph_imgs import write_glyph_imgs_mp
from data_utils.write_data_to_dirs import create_db
from data_utils.relax_rep import relax_rep
from test_few_shot import test_main_model
from options import get_parser_main_model

opts = get_parser_main_model().parse_args()

# Config on opts
# Inference opts
opts.mode = "test"
opts.language = "tha"
opts.char_num = 44
opts.ref_nshot = 8
opts.batch_size = 1 # inference rule
opts.img_size = 64
opts.max_seq_len = 121
opts.name_ckpt = ""
opts.model_path = "./inference_model/950_49452.ckpt"
opts.ref_char_ids = "0,1,2,3,4,5,6,7"
opts.dir_res = "./inference"
opts.data_root = "./inference/vecfont_dataset/"

# Data preprocessing opts
opts.data_path = './inference'
opts.sfd_path = f'{opts.data_path}/font_sfds'
opts.ttf_path = f'{opts.data_path}/font_ttfs'
opts.split = "test"
opts.debug = True # Save Image On write_glyph_imgs_mp
opts.output_path = f'{opts.data_path}/vecfont_dataset/'
opts.phase = 0
opts.FONT_SIZE = 1

opts.streamlit = True

# Glypts ID :
# [(0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E')]
# [(5, 'F'), (6, 'G'), (7, 'H'), (8, 'I'), (9, 'J')]
# [(10, 'K'), (11, 'L'), (12, 'M'), (13, 'N'), (14, 'O')]
# [(15, 'P'), (16, 'Q'), (17, 'R'), (18, 'S'), (19, 'T')]
# [(20, 'U'), (21, 'V'), (22, 'W'), (23, 'X'), (24, 'Y')]
# [(25, 'Z'), (26, 'a'), (27, 'b'), (28, 'c'), (29, 'd')]
# [(30, 'e'), (31, 'f'), (32, 'g'), (33, 'h'), (34, 'i')]
# [(35, 'j'), (36, 'k'), (37, 'l'), (38, 'm'), (39, 'n')]
# [(40, 'o'), (41, 'p'), (42, 'q'), (43, 'r'), (44, 's')]
# [(45, 't'), (46, 'u'), (47, 'v'), (48, 'w'), (49, 'x')]
# [(50, 'y'), (51, 'z'), (52, 'ก'), (53, 'ข'), (54, 'ฃ')]
# [(55, 'ค'), (56, 'ฅ'), (57, 'ฆ'), (58, 'ง'), (59, 'จ')]
# [(60, 'ฉ'), (61, 'ช'), (62, 'ซ'), (63, 'ฌ'), (64, 'ญ')]
# [(65, 'ฎ'), (66, 'ฏ'), (67, 'ฐ'), (68, 'ฑ'), (69, 'ฒ')]
# [(70, 'ณ'), (71, 'ด'), (72, 'ต'), (73, 'ถ'), (74, 'ท')]
# [(75, 'ธ'), (76, 'น'), (77, 'บ'), (78, 'ป'), (79, 'ผ')]
# [(80, 'ฝ'), (81, 'พ'), (82, 'ฟ'), (83, 'ภ'), (84, 'ม')]
# [(85, 'ย'), (86, 'ร'), (87, 'ล'), (88, 'ว'), (89, 'ศ')]
# [(90, 'ษ'), (91, 'ส'), (92, 'ห'), (93, 'ฬ'), (94, 'อ')]
# [(95, 'ฮ')]

import string 
import pythainlp

thai_digits = [*pythainlp.thai_digits]
thai_characters = [*pythainlp.thai_consonants]
eng_characters = [*string.ascii_letters]
thai_floating = [*pythainlp.thai_vowels]

directories = [
    "inference",
    "inference/char_set",
    "inference/font_sfds",
    "inference/font_ttfs",
    "inference/vecfont_dataset",
    "inference/font_ttfs/tha/test",
    ]


# Data Preprocessing
def preprocessing(ttf_file) -> str:
    shutil.rmtree("inference") 
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Save File / Copy File
    if isinstance(ttf_file, memoryview):
        with open(f"{opts.data_path}/font_ttfs/tha/test/0000.ttf", 'wb') as f:
            f.write(ttf_file)
    elif isinstance(ttf_file, str):
        shutil.copy(ttf_file, f"{opts.data_path}/font_ttfs/tha/test/0000.ttf")

    glypts = sorted(set(thai_characters))
    print("Glypts:",len(glypts))
    print("".join(glypts))
    f = open("inference/char_set/tha.txt", "w")
    f.write("".join(glypts))
    f.close()

    # Preprocess Pipeline
    convert_mp(opts)
    write_glyph_imgs_mp(opts)
    output_path = os.path.join(opts.output_path, opts.language, opts.split)
    log_path = os.path.join(opts.sfd_path, opts.language, 'log')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    create_db(opts, output_path, log_path)
    relax_rep(opts)

    print("Finished making a data", ttf_file)
    print("Saved at", output_path)
    return output_path

def inference_model(n_samples, ref_char_ids, version):
    opts.n_samples = n_samples
    opts.ref_char_ids = ref_char_ids

    # Select Model
    if version == "TH2TH":
        opts.model_path = "./inference_model/950_49452.ckpt"
    elif version == "ENG2TH":
        opts.model_path = "./inference_model/950_49452.ckpt"
    else:
        raise NotImplementedError

    return test_main_model(opts)

def ttf_to_image(ttf_file, n_samples=10, ref_char_ids="1,2,3,4,5,6,7,8", version="TH2TH"):
    preprocessing(ttf_file) # Make Data
    merge_svg_img = inference_model(n_samples, ref_char_ids, version) # Inference
    return merge_svg_img

def main():
    print(opts.mode)
    ttf_to_image("font_sample/SaoChingcha-Regular.otf")
    
if __name__ == "__main__":
    main()
