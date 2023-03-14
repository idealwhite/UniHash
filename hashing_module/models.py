import torch
from torch import nn
from torch.nn import functional as F
import torch as t
import time


class BasicModule(nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            self.load_state_dict(t.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), 'hashing_checkpoint/' + name)
        return name

    def forward(self, *input):
        pass


class LinearModel(BasicModule):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(LinearModel, self).__init__()
        #self.module_name = "text_model"
        self.linear = nn.Linear(y_dim, bit)
        self.linear1 = nn.Linear(768, bit)
        self.linear2 = nn.Linear(300, bit)


    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear(x)
        #x = self.linear1(x)
        #x = self.linear2(x)
        #x = self.conv2(x)

        return x
class ClassifyLinearModel(BasicModule):
    def __init__(self, bit, num_class):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ClassifyLinearModel, self).__init__()
        #self.module_name = "text_model"
        self.linear = nn.Linear(bit, 200)
        self.linear1 = nn.Linear(200, 100)
        self.linear2 = nn.Linear(100, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear(x)
        x = self.linear1(x)
        x = self.linear2(x)
        #x = self.conv2(x)
        x = self.sigmoid(x)
        return x
class ClassifyLinearModelForConstractive(BasicModule):
    def __init__(self, hidden=768, num_class=24):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ClassifyLinearModelForConstractive, self).__init__()
        #self.module_name = "text_model"
        self.linear = nn.Linear(hidden, 200)
        self.linear1 = nn.Linear(200, 100)
        self.linear2 = nn.Linear(100, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear(x)
        x = self.linear1(x)
        x = self.linear2(x)
        #x = self.conv2(x)
        x = self.sigmoid(x)
        return x
class ClassifyLinearModel(BasicModule):
    def __init__(self, hidden=64, num_class=24):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ClassifyLinearModel, self).__init__()
        #self.module_name = "text_model"
        self.linear = nn.Linear(hidden, num_class)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear(x)
        return x