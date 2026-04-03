# dataset
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import torchattacks
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super(WrappedModel, self).__init__()
        self.base_model = base_model
        # 预训练模型需要的归一化层
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        # 先归一化再送入模型
        return self.base_model(self.norm(x))


def im_convert(tensor):
    """
    将 [C, H, W] 的 Tensor 转为 [H, W, C] 的 Numpy Array，
    并确保数值在 [0, 1] 之间。
    """
    image = tensor.cpu().clone().detach().numpy()  # [C, H, W]
    image = image.transpose(1, 2, 0)  # 维度转置，得到 [H, W, C]
    image = image.clip(0, 1)  # 裁剪，防止数值超出 [0, 1] 范围
    return image


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def un_normalize(tensor, mean, std):
    """将归一化后的数据反转回 [0, 1]"""
    # 这里处理的是单个样本 [C, H, W]
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    # 反归一化公式: original = (normalized * std) + mean
    image = (image * std) + mean
    image = image.clip(0, 1)  # 确保裁剪回合理范围
    return image


def show_attack_comparison(org_tensor, adv_tensor, org_label, adv_pred, class_names):
    """
    显示原始图片、噪声图、对抗图片以及预测结果。
    """
    # 1. 转换原始图
    org_img = im_convert(org_tensor)
    # 2. 转换对抗图
    adv_img = im_convert(adv_tensor)
    # 3. 计算噪声图 (噪声 = 对抗 - 原始)
    noise = adv_img - org_img
    # 噪声通常很小，为了看清需要做归一化增强
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    # 4. 绘图
    plt.figure(figsize=(15, 5))

    # 子图1：原始图片
    plt.subplot(1, 3, 1)
    plt.imshow(org_img)
    plt.title(f"Original: {class_names[org_label]} ")
    plt.axis('off')

    # 子图2：噪声 (幅度放大后)
    plt.subplot(1, 3, 2)
    plt.imshow(noise, cmap='gray' if noise.shape[2] == 1 else None)
    plt.title("Adversarial Noise\n(Scaled for Visibility)")
    plt.axis('off')

    # 子图3：对抗样本
    plt.subplot(1, 3, 3)
    plt.imshow(adv_img)
    plt.title(f"Adversarial: {class_names[adv_pred]} \nSuccess!")
    plt.axis('off')

    plt.tight_layout()
    plt.show()  # 在界面上显示图片


# 1. 修正后的数据准备
# 增加 Resize 使其适配 ResNet 预训练模型
tran = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=tran, download=True)
indices = [i for i, label in enumerate(dataset.targets) if label == 3]
subset_cat = Subset(dataset, indices)
# 使用 DataLoader 方便一次取 10 张
loader = DataLoader(subset_cat, batch_size=10, shuffle=False)
original_img, _ = next(iter(loader))
original_img = original_img.to(device)

# 2. 修正后的模型加载
model = models.resnet50(pretrained=True).to(device)
model.eval()
final_model = WrappedModel(model).to(device)
final_model.eval()  # 确保在评估模式

# 3. 修正归一化逻辑
# 将归一化整合进一个临时函数，让攻击器知道归一化的存在
norm_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 核心修正：攻击这个“带归一化”的 lambda 模型
atk = torchattacks.PGD(final_model, eps=8/255, alpha=2/255, steps=10)
atk.set_mode_targeted_by_label()

# 4. 生成攻击（确保 target_labels 和 original_img 数量一致）
target_labels = torch.full((original_img.size(0),), 0).long().to(device)
atk_img = atk(original_img, target_labels)

# 5. 预测结果分析 (由于 final_model 已经内置了归一化，直接喂图即可)
with torch.no_grad():
    atk_outputs = final_model(atk_img)
    atk_preds = torch.argmax(atk_outputs, dim=1)

# 6. 可视化
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 临时修改打印逻辑，避免索引越界
for i in range(len(atk_preds)):
    pred_idx = atk_preds[i].item()
    # 只有当索引在 0-9 之间时才去查 classes 表
    name = classes[pred_idx] if pred_idx < 10 else f"ImageNet类({pred_idx})"
    print(f"样本 {i} - 预测结果: {name}")

    # 修正 show_attack_comparison 的调用，不要传入越界的索引
    # 这里我们强制传 0 或其他数字，仅为了让绘图函数不崩溃
    safe_pred = pred_idx if pred_idx < 10 else 0
    show_attack_comparison(original_img[i], atk_img[i], 3, safe_pred, classes)