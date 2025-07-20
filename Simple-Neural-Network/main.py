# ----------------------------------------------------------
# 生成训练数据
# ----------------------------------------------------------
import numpy as np
import random

# 定义真实的权重向量（训练目标）
w_list = np.array([2, 3, 4, 7, 11, 5, 13])

# 生成10个随机输入样本（每个样本长度与w_list相同）
x_list = []
for _ in range(10):
    x_sample = np.array([random.randint(1, 100) for _ in range(len(w_list))])
    x_list.append(x_sample)

# 计算对应的目标输出（输入与真实权重的点积）
y_list = []
for x_sample in x_list:
    y_temp = x_sample @ w_list  # 矩阵乘法计算预测值
    y_list.append(y_temp)

# -----------------------------------------------------------
# 准备训练环境
# -----------------------------------------------------------
import torch 
import torch.nn as nn

# 定义自定义线性模型
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化可学习参数（长度与w_list相同）
        self.w = nn.Parameter(torch.randn(len(w_list), dtype=torch.float32))
        print("初始权重:", self.w)  # 打印初始化的权重
    
    def forward(self, x: torch.Tensor):
        # 前向传播：权重向量与输入向量的点积
        return self.w @ x

# 实例化模型（CPU模式）
model = MyLinear()  # 不使用GPU

# 定义损失函数（均方误差）
loss_fn = nn.MSELoss()

# 定义优化器（随机梯度下降，学习率0.00001）
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

# 数据预处理（CPU模式）
x_input = torch.tensor(x_list, dtype=torch.float32)  # 输入数据转为张量
y_output = torch.tensor(y_list, dtype=torch.float32)  # 目标数据转为张量

# ------------------------------------------------------------
# 开始模型训练
# ------------------------------------------------------------
num_epochs = 200  # 总训练轮数

for epoch in range(num_epochs):
    for i, x in enumerate(x_input):
        # 前向传播
        y_pred = model(x)  # 获取模型预测值
        
        # 计算损失
        loss = loss_fn(y_pred, y_output[i])  # 比较预测值与真实值
        
        # 清除梯度缓存
        optimizer.zero_grad()  # 清除之前的梯度信息
        
        # 反向传播
        loss.backward()  # 计算当前参数的梯度
        
        # 参数更新
        optimizer.step()  # 根据梯度更新模型参数
    
    # 每10轮打印一次训练进度
    if (epoch+1) % 10 == 0:
        print(f'轮次 [{epoch+1}/{num_epochs}], 损失: {loss.item():.4f}')

print("训练完成")
print("最终权重:", model.w.detach().numpy())


