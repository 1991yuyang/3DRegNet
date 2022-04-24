import torch as t
from torch import nn
from torch.nn import functional as F


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

        :param x: shape like (N, C, F), N represents the number of point correspondence set,
                  C represents the number of point correspondence of one point correspondence set, F represents the in_features,
                  in_features is 6 if x is original point cloud
        :return: shape like (N, C, out_features)
        """
        return self.block(x)


class Res(nn.Module):

    def __init__(self):
        super(Res, self).__init__()
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    d = t.randn(2, 3, 256)
    model = FC(in_features=256, out_features=512, is_relu=True, is_bn=False, num_of_correspondence=3)
    out = model(d)
    print(out.size())
