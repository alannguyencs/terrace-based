import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch.nn.functional as F
from PIL import Image
from shutil import rmtree
import math
import queue
from shutil import copyfile
import sys
import platform
from random import shuffle
from tqdm import trange, tqdm
import colorsys
from pathlib import Path
from collections import defaultdict, OrderedDict
import json
import glob
from utils import util_os

project_path = "/home/tnguyenhu2/alan_project/"
DATA_PATH = project_path + '/data/foodcounting/'
TERRACE_PATH = project_path + "terrace/"

CKPT_PATH = util_os.gen_dir(TERRACE_PATH + 'ckpt/')
RESULT_PATH = util_os.gen_dir(TERRACE_PATH + 'result/')
LOG_PATH = util_os.gen_dir(TERRACE_PATH + 'log/')

INPUT_SIZE = 256 
SEG_OUTPUT_SIZE = 256

IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]


FIVE_CONTOUR_POLYGON = 'five_contour_polygon'
CONTOUR_INTENSITY_SCALE = 51


COUNTING_DICT = {'cookie': 9, 'dimsum': 6, 'sushi': 6}
MASK_TYPES = [FIVE_CONTOUR_POLYGON]   #PNG



#CUDA ENVIRONMENT
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
