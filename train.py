import torch as t
from torch import nn, optim
import os
from model import RefineNet
from loss import RefineLoss
from dataLoader import make_loader, find_feature_dist_thresh
from metric import RotMatMetric, TransMetric
CUDA_VISIBLE_DEVICES = "0"
device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


def train_epoch(model, criterion, train_loader, optimizer):
    pass


def valid_epoch(model, criterion, valid_loader):
    pass

