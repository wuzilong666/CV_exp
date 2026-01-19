import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import os
import cv2
import numpy as np
from pathlib import Path

class Net(nn.Module):
    """基于MNIST的轻量级卷积网络。"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tloss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\n测试集: 平均损失: {test_loss:.4f}, '
          f'准确率: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

def predict_student_id(model, device, image_path):
    if not os.path.exists(image_path):
        print(f"未找到图片 {image_path}。")
        return

    print(f"正在处理图片: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print("加载图片失败。")
        return

    # 创建输出图片副本
    output_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积过滤轮廓
    min_area = 20
    digit_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # 从左到右排序轮廓
    digit_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    predicted_id = ""
    model.eval()
    for c in digit_contours:
        x, y, w, h = cv2.boundingRect(c)

        # 提取感兴趣区域 (ROI)
        roi = thresh[y:y+h, x:x+w]

        # 调整大小为 28x28 并填充以保持纵横比
        target_size = 20
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            continue

        resized_roi = cv2.resize(roi, (new_w, new_h))

        # 放置在 28x28 画布的中心
        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_roi

        # 归一化
        tensor_img = transforms.ToTensor()(canvas)
        tensor_img = transforms.Normalize((0.1307,), (0.3081,))(tensor_img)
        tensor_img = tensor_img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            pred = output.argmax(dim=1, keepdim=True)
            digit = pred.item()
            predicted_id += str(digit)

        # 在输出图片上绘制矩形和预测结果
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_img, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)

    print(f"预测学号: {predicted_id}")

    # 保存结果图片
    image_dir = Path(image_path).parent
    output_path = image_dir / 'exp3_result' / 'result_prediction_myID.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output_img)
    print(f"结果图片已保存为 '{output_path}'")

    return predicted_id

def get_device():
    """选择并打印当前训练设备."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

def build_dataloader(data_root, batch_size, use_cuda):
    """封装数据预处理与 DataLoader 构建."""
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if not data_root.exists():
        raise FileNotFoundError(f"数据集路径不存在: {data_root}\n")
    dataset = datasets.ImageFolder(root=str(data_root), transform=train_transform)
    print(f"数据集大小: {len(dataset)} 张图片")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1 if use_cuda else 0,
        pin_memory=use_cuda
    )

def load_or_initialize_model(model, save_path, device, train_loader, epochs, lr, gamma):
    """负责加载现有模型或重新训练并保存."""
    if save_path.exists():
        print(f"已有模型，正在从 {save_path} 加载...")
        model.load_state_dict(torch.load(str(save_path), map_location=device))
        print("模型加载成功！")
        return model
    print("未找到模型，开始训练...")
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    print(f"模型已保存至 {save_path}")
    return model

def main():
    script_dir = Path(__file__).parent.resolve()
    device = get_device()
    use_cuda = device.type == "cuda"
    batch_size, epochs, lr, gamma = 64, 5, 1.0, 0.7
    data_root = script_dir / 'minist_dataset' / 'training'
    train_loader = build_dataloader(data_root, batch_size, use_cuda)
    save_path = script_dir / "mnist_model.pth"
    model = load_or_initialize_model(
        Net().to(device),
        save_path,
        device,
        train_loader,
        epochs,
        lr,
        gamma
    )
    predict_student_id(model, device, str(script_dir / "myID.png"))

if __name__ == '__main__':
    main()