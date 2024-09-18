import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Grayscale

# 将根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gans import Generator, Discriminator  # 确保 gans.py 中定义了这些类

# 图像转换设置
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    Grayscale(num_output_channels=1),  # 添加此行将图像转换为灰度
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 用于灰度图像的归一化
])

# 加载数据集
dataset = datasets.ImageFolder(root='./train1', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# 训练循环
for epoch in range(100):  # 设置训练轮数
    for images, _ in train_loader:
        real_data = images.cuda()
        batch_size = real_data.size(0)

        # 真实图像的标签
        real_labels = torch.ones(batch_size, 1).cuda()
        fake_labels = torch.zeros(batch_size, 1).cuda()

        # 训练判别器
        discriminator.zero_grad()
        outputs_real = discriminator(real_data)
        loss_real = criterion(outputs_real, real_labels)

        # 生成假图像
        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake_data = generator(noise)
        outputs_fake = discriminator(fake_data.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        # 反向传播和优化
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # 训练生成器
        generator.zero_grad()
        outputs_fake = discriminator(fake_data)
        loss_g = criterion(outputs_fake, real_labels)
        loss_g.backward()
        optimizer_g.step()

    print(f'Epoch {epoch+1}, Loss_D: {loss_d.item()}, Loss_G: {loss_g.item()}')

# 保存模型
os.makedirs('models', exist_ok=True)
torch.save(generator.state_dict(), 'models/generator.pth')
torch.save(discriminator.state_dict(), 'models/discriminator.pth')
