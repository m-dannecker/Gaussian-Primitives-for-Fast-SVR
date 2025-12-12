import os
import sys
import time
import random
import argparse
import yaml
import shutil
import warnings
import subprocess  # <--- NEW IMPORT
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import faiss
import faiss.contrib.torch_utils
import ants
from mpmath import si
from sympy import Q
import scipy.ndimage

from gaussian_primitives_svr import train

# --- PREAMBLE ---
warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")
torch.set_float32_matmul_precision('high')

# --- MAIN ---
def main(config_path):
    parser = argparse.ArgumentParser()
    # If called via subprocess with --config, that value takes precedence.
    # If called manually with a function argument, the default is used.
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--run_svort', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default=None)
    
    # Use parse_known_args to ensure flexibility if extra flags are ever passed
    args, _ = parser.parse_known_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    for i, subject in enumerate(cfg['data']['subjects']):
        if not subject['enabled']: continue
        print(f"--- Starting Subject {i+1} of {len(cfg['data']['subjects'])} ---")
        exp_name = args.exp_name if args.exp_name else cfg['data']['subjects'][i]['name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_name = f"{exp_name}_{timestamp}.nii.gz"
        # cwd = os.getcwd()
        output_file_path = os.path.join(cfg['experiment']['output_root'], exp_name, output_file_name)
        output_file_path = os.path.abspath(output_file_path)
        print(f"Output file path: {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path.replace('.nii.gz', '.yaml'), 'w') as f: yaml.dump(cfg, f)

        stack_paths = cfg['data']['subjects'][i]['input_stacks']
        mask_paths = cfg['data']['subjects'][i]['input_masks']
        if not mask_paths: mask_paths = []*len(stack_paths)

        try:
            train(stack_paths=stack_paths, mask_paths=mask_paths, config=cfg, output_file_path=output_file_path)
        except Exception as e:
            print(f"Error during training: {e}")
            print(f"Stack paths: {stack_paths}")
            print(f"Mask paths: {mask_paths}")
            print(f"Config: {cfg}")
            print(f"Output file path: {output_file_path}")
            raise e


if __name__ == "__main__":
    # The script acts in two modes:
    # 1. Master Mode: No arguments (or different args) -> Runs run_ablation_study()
    # 2. Worker Mode: Called with --config -> Runs main()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    # Add other arguments purely to prevent argparse errors if they are passed
    parser.add_argument('--run_svort', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default=None)
    
    args, unknown = parser.parse_known_args()

    main(args.config)

