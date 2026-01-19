import os
import sys
import math
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    """支持中文/Unicode 路径的图像读取。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """支持中文/Unicode 路径的图像保存。"""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
        path = path + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def to_gray(bgr_img: np.ndarray) -> np.ndarray:
    """手写 BGR 转灰度，输出 float32 灰度图。"""
    h, w, _ = bgr_img.shape
    gray = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            b, g, r = bgr_img[i, j]
            gray[i, j] = 0.114 * b + 0.587 * g + 0.299 * r
    return gray


def pad_image(img: np.ndarray, pad: int) -> np.ndarray:
    """对灰度图进行零填充，便于卷积。"""
    h, w = img.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float32)
    padded[pad:pad + h, pad:pad + w] = img
    return padded


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """手写 2D 卷积，不调用现成滤波函数。"""
    kh, kw = kernel.shape
    pad = kh // 2
    padded = pad_image(img, pad)
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * kernel)
    return out


def sobel_filter(gray: np.ndarray) -> np.ndarray:
    """应用 Sobel 水平/垂直核，输出梯度幅值图."""
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = conv2d(gray, kx)
    gy = conv2d(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = np.clip(mag, 0, 255)
    return mag.astype(np.uint8)


def custom_kernel_filter(gray: np.ndarray) -> np.ndarray:
    """应用给定卷积核进行滤波."""
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32)
    resp = conv2d(gray, kernel)
    resp = np.clip(resp, 0, 255)
    return resp.astype(np.uint8)


def color_histogram(img: np.ndarray) -> np.ndarray:
    """逐像素统计 B/G/R 三通道直方图，shape 为 (3,256)。"""
    hist = np.zeros((3, 256), dtype=np.int64)
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            hist[0, b] += 1
            hist[1, g] += 1
            hist[2, r] += 1
    return hist


def gray_comatrix(gray: np.ndarray, levels: int = 256, dx: int = 1, dy: int = 0) -> np.ndarray:
    """构建灰度共生矩阵（默认右邻居），返回归一化矩阵."""
    h, w = gray.shape
    mat = np.zeros((levels, levels), dtype=np.float64)
    for i in range(h):
        ny = i + dy
        if ny < 0 or ny >= h:
            continue
        for j in range(w):
            nx = j + dx
            if nx < 0 or nx >= w:
                continue
            g1 = int(gray[i, j])
            g2 = int(gray[ny, nx])
            mat[g1, g2] += 1
    total = np.sum(mat)
    if total > 0:
        mat = mat / total
    return mat


def texture_features(glcm: np.ndarray) -> dict:
    """由灰度共生矩阵计算对比度、能量、同质性、熵."""
    contrast = 0.0
    energy = 0.0
    homogeneity = 0.0
    entropy = 0.0
    levels = glcm.shape[0]
    for i in range(levels):
        for j in range(levels):
            p = glcm[i, j]
            if p <= 0:
                continue
            contrast += (i - j) ** 2 * p
            energy += p * p
            homogeneity += p / (1 + abs(i - j))
            entropy -= p * math.log(p, 2)
    return {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "entropy": entropy,
    }


def save_npy(data, path: str):
    """保存数据为 npy 文件."""
    np.save(path, data)


def main():
    """命令行入口，完成滤波、直方图与纹理特征提取."""
    parser = argparse.ArgumentParser(description="Sobel 与自定义卷积滤波，颜色直方图和纹理特征提取")
    parser.add_argument("--input", default="exp1_picture.png", help="输入图像路径")
    # 默认输出到脚本同级目录下的 result 文件夹
    parser.add_argument("--outdir", default="result", help="输出目录（默认为脚本同级的 result 文件夹）")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, input_path)
    # 若默认目录不存在该文件，回退到上一级目录尝试
    if not os.path.exists(input_path):
        fallback_path = os.path.join(os.path.dirname(script_dir), args.input)
        if os.path.exists(fallback_path):
            input_path = fallback_path

    # 统一处理输出目录：相对路径时基于脚本目录
    outdir_path = args.outdir
    if not os.path.isabs(outdir_path):
        outdir_path = os.path.join(script_dir, outdir_path)
    os.makedirs(outdir_path, exist_ok=True)

    if not os.path.exists(input_path):
        print("读取失败：文件不存在:", input_path)
        sys.exit(1)

    img = imread_unicode(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print("读取失败：文件存在但无法解码（可能损坏/非图片/权限问题）:", input_path)
        sys.exit(1)

    gray = to_gray(img)
    sobel_img = sobel_filter(gray)
    custom_img = custom_kernel_filter(gray)

    # 保存滤波结果（使用支持中文路径的写函数）
    sobel_path = os.path.join(outdir_path, "sobel.png")
    custom_path = os.path.join(outdir_path, "custom_kernel.png")
    if not imwrite_unicode(sobel_path, sobel_img):
        print("保存失败:", sobel_path)
    if not imwrite_unicode(custom_path, custom_img):
        print("保存失败:", custom_path)

    # 可视化颜色直方图
    hist = color_histogram(img)
    plt.figure(figsize=(6, 4))
    colors = ["b", "g", "r"]
    bins = np.arange(256)
    for idx, c in enumerate(colors):
        plt.plot(bins, hist[idx], color=c, label=f"{c.upper()} channel")
    plt.xlim(0, 255)
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_path, "color_hist.png"), dpi=150)
    plt.close()

    # 纹理特征
    glcm = gray_comatrix(gray.astype(np.uint8))
    feats = texture_features(glcm)
    save_npy(feats, os.path.join(outdir_path, "texture_feats.npy"))

    print("完成，输出目录:", outdir_path)


if __name__ == "__main__":
    main()
