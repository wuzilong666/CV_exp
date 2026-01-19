import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os

# 设置清华镜像源加速模型下载
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_cache')
torch.hub.set_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_cache'))

# 使用清华源（设置环境变量）
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# 设置torchvision使用清华源下载预训练权重
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
# 手动指定权重URL为清华源
TSINGHUA_TORCH_URL = "https://mirrors.tuna.tsinghua.edu.cn/pytorch-vision/models/"

# COCO数据集类别名称（部分）
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck'
}

# 自行车在COCO数据集中的类别ID
BICYCLE_CLASS_ID = 2

# 加载预训练的Faster R-CNN模型
def load_model():
    print("正在加载预训练模型")
    
    # 直接从清华源下载权重文件
    import urllib.request
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    weight_file = os.path.join(cache_dir, 'fasterrcnn_resnet50_fpn_coco.pth')
    
    if not os.path.exists(weight_file):
        print("从清华源下载预训练权重...")
        url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
        try:
            torch.hub.download_url_to_file(url, weight_file)
        except Exception as e:
            print(f"下载失败: {e}")
            print("尝试使用默认方式下载...")
            weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            model = fasterrcnn_resnet50_fpn(weights=weights)
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model, device
    
    # 加载本地权重
    model = fasterrcnn_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"模型加载完成.")
    return model, device

# 检测图片中的自行车
def detect_bicycle(model, device, image_path, confidence_threshold=0.8):
    image = Image.open(image_path).convert('RGB')
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 进行检测
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 解析检测结果
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # 筛选自行车检测结果
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if label == BICYCLE_CLASS_ID and score >= confidence_threshold:
            detections.append((box, score))
            print(f"检测到自行车，置信度: {score:.2f}, 位置: {box}")
    
    return image, detections

# 在图片上绘制检测结果
def draw_results(image, detections, output_path):
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    for box, score in detections:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=7)
        label = f"bicycle: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(text_bbox, fill='red')
        draw.text((x1, y1 - 25), label, fill='white', font=font)
    
    # 保存结果
    image.save(output_path)
    print(f"结果已保存至: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "bicycle1.png")
    output_path = os.path.join(script_dir, "bicycle1_result.png")

    if not os.path.exists(input_path):
        print(f"错误：输入图片不存在: {input_path}")
        return
    
    # 加载模型
    model, device = load_model()

    print(f"正在检测图片: {input_path}")
    image, detections = detect_bicycle(model, device, input_path, confidence_threshold=0.7)
    
    if len(detections) == 0:
        print("未检测到自行车，尝试降低置信度阈值...")
        image, detections = detect_bicycle(model, device, input_path, confidence_threshold=0.3)
    if len(detections) == 0:
        print("仍未检测到自行车")
    else:
        print(f"共检测到 {len(detections)} 辆自行车")
    
    # 绘制并保存结果
    draw_results(image, detections, output_path)


if __name__ == "__main__":
    main()
