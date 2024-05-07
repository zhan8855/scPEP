import numpy
import pandas
import torch
import random
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_name", default="norman")
    parser.add_argument("--data_path", default="/path/to/data/dir", type=str)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_output(file_name):
    file = open(file_name)
    lines = file.readlines()
    results = {}
    for line in lines:
        if line.startswith("test"):
            key, val = line.strip().split(": ")
            if key not in results:
                results[key] = [val]
            else:
                results[key].append(val)
    out_file_name = ".".join(file_name.split(".")[:-1]) + ".csv"
    pandas.DataFrame(results).to_csv(out_file_name)