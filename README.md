基于torchattaks的PGD算法误导resnet
这是一个关于深度学习安全性的项目。基于torchattacks的PGD算法，给图片添加特定的微小扰动，误导resNet将猫认成飞机，误导率达到100%，定向误导率达到80%（模型也会把猫认成袜子或蜥蜴）。
环境要求
基于python3.12与pytorch环境
主要依赖库包括：torchattacks、torchvision、numpy、matplotlib
安装命令：pip install torch torchvision torchattacks matplotlib numpy
核心实现
所使用预训练的resnet，其输入经过均值和方差归一化。我们封装了一个WrappedModel类，添加均值和方差归一化的前向操作：
class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super(WrappedModel, self).__init__()
        self.base_model = base_model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.base_model(self.norm(x))


