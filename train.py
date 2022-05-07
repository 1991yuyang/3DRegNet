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


print_step = 1
epoch = 1000
batch_size = 1
lr = 0.0001
lr_de_epoch = 100
lr_de_rate = 0.1
threeDRegNet_count = 2
res_block_counts = [8, 4]
num_of_correspondence = 3000
loss_alpha = 0.001
loss_beta = 0.5
use_lie = True
num_workers = 1
train_data_dir = r"F:\python_project\test_open3d\pcd_dir"
valid_data_dir = r"F:\python_project\test_open3d\pcd_dir"
voxel_size = 0.01
R_range = [-2, 2]
t_range = [-0.02, 0.02]
noise_strength = 0.12
feature_distance_tresh = find_feature_dist_thresh(train_data_dir, voxel_size, R_range, t_range, num_of_correspondence, noise_strength)
M = 3 if use_lie else 9
best_valid_loss = float("inf")


def train_epoch(model, criterion, train_loader, optimizer, current_epoch):
    model.train()
    step = len(train_loader)
    current_step = 1
    for d_train, class_label_train, _, R_train, t_vec_train in train_loader:
        d_train_cuda = d_train.cuda(device_ids[0])
        class_label_train_cuda = class_label_train.cuda(device_ids[0])
        rotation_mats, trans_mats, cls_outs, reg_outs, use_for_cls_losses, points_preds = model(d_train_cuda)
        train_loss = criterion(use_for_cls_losses, class_label_train_cuda, points_preds, d_train_cuda[:, :, 3:])
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        R_met, R_pred = RotMatMetric(rotation_mats, R_train)
        t_met, t_pred = TransMetric(trans_mats, rotation_mats, t_vec_train)
        if current_step % print_step == 0:
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, R_metric:%.5f, t_metric:%.5f" % (current_epoch, epoch, current_step, step, train_loss.item(), R_met.item(), t_met.item()))
        current_step += 1
    print("saving epoch model......")
    t.save(model.state_dict(), "epoch.pth")
    return model


def valid_epoch(model, criterion, valid_loader, current_epoch):
    global best_valid_loss
    model.eval()
    step = len(valid_loader)
    accum_loss = 0
    accum_R_met = 0
    accum_t_met = 0
    for d_valid, class_label_valid, _, R_valid, t_vec_valid in valid_loader:
        d_valid_cuda = d_valid.cuda(device_ids[0])
        class_label_valid_cuda = class_label_valid.cuda(device_ids[0])
        with t.no_grad():
            rotation_mats, trans_mats, cls_outs, reg_outs, use_for_cls_losses, points_preds = model(d_valid_cuda)
            valid_loss = criterion(use_for_cls_losses, class_label_valid_cuda, points_preds, d_valid_cuda[:, :, 3:])
            R_met, R_final = RotMatMetric(rotation_mats, R_valid)
            t_met, t_final = TransMetric(trans_mats, rotation_mats, t_vec_valid)
            accum_loss += valid_loss.item()
            accum_R_met += R_met.item()
            accum_t_met += t_met.item()
    if step == 0:
        step = 1
    avg_loss = accum_loss / step
    avg_R_met = accum_R_met / step
    avg_t_met = accum_t_met / step
    print("##########valid epoch:%d############" % (current_epoch,))
    print("valid_loss:%.5f, R_metic:%.5f, t_metric:%.5f" % (avg_loss, avg_R_met, avg_t_met))
    if avg_loss < best_valid_loss:
        best_valid_loss = avg_loss
        print("saving best model......")
        t.save(model.state_dict(), "best.pth")
    return model


def main():
    model = RefineNet(threeDRegNet_count, res_block_counts, num_of_correspondence, M, use_lie)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = RefineLoss(loss_alpha, loss_beta).cuda(device_ids[0])
    lr_sch = optim.lr_scheduler.StepLR(optimizer, lr_de_epoch, lr_de_rate)
    for e in range(epoch):
        current_epoch = e + 1
        train_loader = make_loader(train_data_dir, voxel_size, R_range, t_range, num_of_correspondence, noise_strength, feature_distance_tresh, batch_size, num_workers)
        valid_loader = make_loader(valid_data_dir, voxel_size, R_range, t_range, num_of_correspondence, noise_strength, feature_distance_tresh, batch_size, num_workers)
        model = train_epoch(model, criterion, train_loader, optimizer, current_epoch)
        model = valid_epoch(model, criterion, valid_loader, current_epoch)
        lr_sch.step()


if __name__ == "__main__":
    main()


