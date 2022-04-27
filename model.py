import torch as t
from torch import nn
from resnet import resnet18


class FC(nn.Module):

    def __init__(self, in_features, out_features, is_relu, is_bn, num_of_correspondence):
        """

        :param in_features: number of features of input data
        :param out_features: number of features of output data
        :param is_relu: True use ReLU, False not use
        :param is_bn: True use BatchNorm1d, False not use
        :param num_of_correspondence: number of point correspondence of one point correspondence set
        """
        super(FC, self).__init__()
        is_bias = not is_bn
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=is_bias)
        )
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm1d(num_features=num_of_correspondence))
        if is_relu:
            self.block.add_module("relu", nn.ReLU())

    def forward(self, x):
        """

        :param x: shape like (N, C, F), N represents the number of point correspondence set or Batch Size,
                  C represents the number of point correspondence of one point correspondence set, F represents the in_features,
                  in_features is 6 if x is original point cloud
        :return: shape like (N, C, out_features)
        """
        return self.block(x)


class Res(nn.Module):

    def __init__(self, resblock_count):
        """

        :param resblock_count: number of resnet block
        """
        super(Res, self).__init__()
        self.res_blocks = nn.Sequential()
        for i in range(resblock_count):
            self.res_blocks.add_module("res_%d" % (i,), nn.Sequential(*list(resnet18().children())[:-2]))

    def forward(self, x):
        """

        :param x: shape like (N, C, F), N represents the number of point correspondence set or Batch Size,
                  C represents the number of point correspondence of one point correspondence set, F represents the in_features,
                  in_features is 6 if x is original point cloud
        :return: shape like (N, C, out_features)
        """
        res_results = []  # shape of item of the list is [N, C, F]
        x = x.unsqueeze(1)  # (N, 1, C, F)
        for n, m in self.res_blocks._modules.items():
            x = m(x)
            res_results.append(x.squeeze(1))
        return res_results


class CLSNet(nn.Module):

    def __init__(self, res_block_count, num_of_correspondence):
        """

        :param res_block_count: number of resnet block
        :param num_of_correspondence: number of point correspondence of one point correspondence set
        """
        super(CLSNet, self).__init__()
        self.fc1 = FC(in_features=6, out_features=128, is_relu=True, is_bn=False, num_of_correspondence=num_of_correspondence)
        self.res = Res(res_block_count)
        self.fc2 = FC(in_features=128, out_features=1, is_relu=False, is_bn=False, num_of_correspondence=num_of_correspondence)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        fc1_result = self.fc1(x)
        res_results = self.res(fc1_result)
        use_for_cls_loss = self.fc2(res_results[-1]).squeeze(2)  # shape like (N, C)
        relu_result = self.relu(use_for_cls_loss)
        out = self.tanh(relu_result)  # shape like (N, C)
        cls_features = [fc1_result] + res_results
        return out, cls_features, use_for_cls_loss


class ContextBN(nn.Module):

    def __init__(self):
        super(ContextBN, self).__init__()
        pass

    def forward(self, x):
        x = (x - t.mean(x, dim=1, keepdim=True)) / t.std(x, dim=1, keepdim=True)
        return x


class RegNet(nn.Module):

    def __init__(self, M, res_block_count):
        super(RegNet, self).__init__()
        self.context_bn = ContextBN()
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=(2, 1), padding=(1, 1))
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=8 * ((res_block_count + 1) // 2) * 128, out_features=256),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(in_features=256, out_features=M + 3)

    def forward(self, cls_features):
        pool_results = []
        for cls_feature in cls_features:
            max_pool_result = t.max(cls_feature, dim=1).values  # shape like: (N, F)
            context_bn_result = self.context_bn(max_pool_result)
            pool_results.append(context_bn_result)
        concate_results = t.cat(pool_results, dim=1)
        concate_results = concate_results.view((concate_results.size()[0], len(pool_results), -1))  # shape like: (N, res_block_count + 1, F)
        concate_results = concate_results.unsqueeze(1)  # (N, 1, res_block_count + 1, F)
        conv_result = self.conv(concate_results).view((concate_results.size()[0], -1))  # (N, 8, (res_block_count + 1) // 2, F)
        linear1_result = self.linear1(conv_result)
        reg_result = self.linear2(linear1_result)
        return reg_result


class ThreeDRegNet(nn.Module):

    def __init__(self, res_block_count, num_of_correspondence, M):
        """

        :param res_block_count: resnet block count of Registration Block
        :param num_of_correspondence: number of correspondence of one point correspondence set
        :param M: number of rotation parameter
        """
        super(ThreeDRegNet, self).__init__()
        self.cls = CLSNet(res_block_count, num_of_correspondence)
        self.reg = RegNet(M, res_block_count)

    def forward(self, x):
        """

        :param x: shape like (N, num_of_correspondence, 6), note that x[:, :, :3] is the point set which is registrated
        :return:
        """
        cls_out, cls_features, use_for_cls_loss = self.cls(x)  # cls_out: (N, num_of_correspondence)
        reg_out = self.reg(cls_features)  # reg_out: (N, M + 3)
        return cls_out, reg_out, use_for_cls_loss


class RefineNet(nn.Module):

    def __init__(self, threeDRegNet_count, res_block_count, num_of_correspondence, M):
        super(RefineNet, self).__init__()
        self.M = M
        self.block = nn.Sequential()
        for i in range(threeDRegNet_count):
            self.block.add_module("regnet_%d" % (i,), ThreeDRegNet(res_block_count, num_of_correspondence, M))

    def forward(self, x):
        cls_outs = []  # shape of item is (N, num_of_correspondence)
        reg_outs = []  # shape of item is (N, M + 3)
        use_for_cls_losses = []  # shape of item is (N, num_of_correspondence)
        reg_results = []   # shape of item is (N, num_of_correspondence, 3)
        reg_result = x[:, :, :x.size()[2] // 2]  # point after registrate, shape like (N, num_of_correspondence, 3)
        dest = x[:, :, x.size()[2] // 2:]  # target point of registration, shape like (N, num_of_correspondence, 3)

        for n, m in self.block._modules.items():
            cls_out, reg_out, use_for_cls_loss = m(x)
            reg_result = registration(reg_out,  reg_result, self.M)
            reg_results.append(reg_result)
            x = t.cat([reg_result, dest], dim=2)
            cls_outs.append(cls_out)
            reg_outs.append(reg_out)
            use_for_cls_losses.append(use_for_cls_loss)
        return cls_outs, reg_outs, use_for_cls_losses, reg_results


def registration(reg_out, point_set, M):
    """

    :param M: number of parameter of rotation, like 9 means 3 * 3 rotation matrix
    :param reg_out: reg_out, shape like (N, M + 3), N is the number of point set
    :param point_set: point set, shape like (N, num_of_correspondence, 3), N is the number of point set, num_of_correspondence is the number of point correspondence of one point set, 3 is one point
    :return:
    """
    rotation_mat = reg_out[:, :M].view((reg_out.size()[0], point_set.size()[2], -1))  # shape like (N, 3, M // 3)
    shift_map = reg_out[:, M:]  # shape like (N, 3)
    rot_result = t.matmul(rotation_mat, point_set.permute(dims=[0, 2, 1])).permute(dims=[0, 2, 1])  # shape like (N, num_of_correspondence, M // 3)
    shift_result = rot_result + shift_map.unsqueeze(1)  # shape like (N, num_of_correspondence, M // 3)
    return shift_result


if __name__ == "__main__":
    d = t.randn(2, 512, 6)
    model = RefineNet(1, 5, 512, 9)
    cls_outs, reg_outs, use_for_cls_losses, reg_results = model(d)
    print(reg_results[0].size())
