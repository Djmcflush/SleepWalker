from pydantic import BaseModel
import torch
from os.path import join as pjoin
import numpy as np
from utils.word_vectorizer import WordVectorizer, POS_enumerator

class OptModel(BaseModel):
    data_root: str = './dataset/HumanML3D'
    motion_dir: str = pjoin(data_root, 'new_joint_vecs')
    text_dir: str = pjoin(data_root, 'texts')
    joints_num: int = 22
    max_motion_length: int = 196
    dataset_name: str = 't2m'
    checkpoints_dir: str = pjoin('checkpoints')
    save_root : str = pjoin(checkpoints_dir, 'Comp_v6_KLD01')
    model_dir = pjoin(save_root, 'model')
    meta_dir = pjoin(save_root, 'meta')
    unit_length: int = 4
    dim_text_hidden: int = 512
    dim_coemb_hidden: int = 512
    dim_motion_hidden: int = 1024
    dim_movement_enc_hidden: int = 512
    gpu_id: int = -1
    dim_movement_latent: int = 512
    dim_pose: int = 263
    dim_movement_dec_hidden: int = 512
    max_motion_length: int = 196
    text_enc_mod: str = 'bigru'
    dim_att_vec: int = 256
    dim_z : int = 128
    n_layers_dec : int = 2
    dim_dec_hidden : int = 512
    which_epoch : str = 'latest'
    dim_word : int = 300
    mean = pjoin(data_root, 'HumanML3D/mean.npy')
    std = pjoin(data_root, 'HumanML3D/std.npy')
    text_file = "./dataset/input_est.txt"
    dim_pos_ohot = len(POS_enumerator)
    text_size = dim_text_hidden * 2
    dim_pri_hidden : int = 512
    n_layers_pri : int = 2
    dim_pos_hidden : int = 1024
    is_train : bool = False

