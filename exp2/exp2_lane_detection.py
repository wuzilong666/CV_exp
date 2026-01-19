import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_lane_keep_largest(image_path):
    """
    车道线检测主函数 - 使用最大连通域筛选算法
    核心思路：通过保留最大连通域来去除噪声，精准提取中间车道线
    """
    # 第一步：图像读取与预处理
    image = cv2.imread(image_path)
    if image is None: return
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 第二步：CLAHE光照均衡化 - 提亮远处较暗的车道线
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_equalized = clahe.apply(gray)
    
    # 第三步：中值滤波去噪 - 去除椒盐噪声，保护边缘
    blurred = cv2.medianBlur(gray_equalized, 5)
    
    # 第四步：二值化
    ret, binary = cv2.threshold(blurred, 203, 255, cv2.THRESH_BINARY)
    
    # 第五步：六边形ROI感兴趣区域掩膜
    mask = np.zeros_like(binary)
    hexagon = np.array([
        [
            (int(width * 0.15), height),            # 左下角
            (int(width * 0.05), int(height * 0.35)), # 左中点
            (int(width * 0.35), int(height * 0.2)),   # 左上角
            (int(width * 0.65), int(height * 0.2)),   # 右上角
            (int(width * 0.95), int(height * 0.35)), # 右中点
            (int(width * 0.85), height)              # 右下角
        ]
    ], dtype=np.int32)
    cv2.fillPoly(mask, hexagon, 255)
    roi_binary = cv2.bitwise_and(binary, mask)
    
    # 第六步：形态学开运算 - 先腐蚀后膨胀，去除细小噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel_open)
    
    # 第七步：垂直方向膨胀 - 连接虚线段，形成连续柱体
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 50))
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)
    
    # 第八步：保留有效连通域（支持多车道线）
    # 过滤掉面积过小的噪声，保留所有可能的车道线区域
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(dilated)
    # 面积阈值：小于此值的连通域被视为噪声
    min_area_threshold = 2000
    
    if len(contours) > 0:
        # 保留所有面积大于阈值的连通域
        for contour in contours:
            if cv2.contourArea(contour) > min_area_threshold:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
    
    # 第九步：Canny边缘检测
    edges = cv2.Canny(clean_mask, 50, 150)
    
    # 第十步：霍夫直线检测
    lines = cv2.HoughLinesP(edges, 
                            rho=1,              # 距离分辨率(像素)
                            theta=np.pi/180,    # 角度分辨率(1度)
                            threshold=20,       # 累加器阈值
                            minLineLength=20,   # 最小线段长度
                            maxLineGap=200)     # 允许间隙
    
    # 第十一步：绘制检测结果
    result_img = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制ROI区域到原图上
    roi_overlay = image.copy()
    cv2.polylines(roi_overlay, [hexagon], True, (0, 255, 255), 3)  # 黄色六边形边框
    roi_filled = image.copy()
    cv2.fillPoly(roi_filled, [hexagon], (0, 255, 255))  # 黄色填充
    roi_visual = cv2.addWeighted(image, 0.7, roi_filled, 0.3, 0)  # 半透明叠加
    cv2.polylines(roi_visual, [hexagon], True, (0, 255, 255), 2)  # 边框

    # 可视化展示关键步骤
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("hexagonal ROI area", fontsize=12)
    plt.imshow(cv2.cvtColor(roi_visual, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Dilated (with noise)", fontsize=12)
    plt.imshow(dilated, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("Keep Largest (denoised)", fontsize=12)
    plt.imshow(clean_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("final result", fontsize=24)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('D:\\a_Work\\3_term1\\CV\\exp\\exp2\\exp2_result.png', dpi=300, bbox_inches='tight')
    plt.show()

process_lane_keep_largest('D:\\a_Work\\3_term1\\CV\\exp\\exp2\\lane.jpg')