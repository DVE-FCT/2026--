import time
import os
import re
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import Common


# ============================================================
# 自动定位最新的 model_X 文件夹
# ============================================================
MODEL_ROOT = "./model"

def get_latest_model_dir():
    if not os.path.exists(MODEL_ROOT):
        raise FileNotFoundError(f"未找到模型目录: {MODEL_ROOT}")
    indices = []
    for name in os.listdir(MODEL_ROOT):
        m = re.match(r'^model_(\d+)$', name)
        if m and os.path.isdir(os.path.join(MODEL_ROOT, name)):
            indices.append(int(m.group(1)))
    if not indices:
        raise FileNotFoundError(f"未找到任何 model_X 文件夹: {MODEL_ROOT}")
    latest_idx = max(indices)
    return os.path.join(MODEL_ROOT, f"model_{latest_idx}"), latest_idx


# ============================================================
# 单张图片推理
# ============================================================
def pridect(imagePath, modelPath, trueLabel=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    t0 = time.time()
    image = Image.open(imagePath).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(Common.device)
    preprocess_time = time.time() - t0

    from model import model as weatherModel
    model = weatherModel
    model.load_state_dict(torch.load(modelPath, map_location=Common.device))
    model = model.to(Common.device)
    model.eval()

    t1 = time.time()
    with torch.no_grad():
        output = model(image_tensor)
    if Common.device.type == 'cuda':
        torch.cuda.synchronize()
    infer_time = time.time() - t1

    t2 = time.time()
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, dim=1)
    conf = conf.item()
    pred = pred.item()
    post_time = time.time() - t2

    total = preprocess_time + infer_time + post_time

    print()
    print(f"真实标签：{trueLabel if trueLabel else '未知'}")
    print(f"预测结果：{Common.labels[pred]}")
    print(f"置信度  ：{conf:.4f} ({conf*100:.2f}%)")
    print(f"耗时统计（总计 {total*1000:.1f}ms）:")
    print(f"  图片预处理: {preprocess_time*1000:.1f}ms")
    print(f"  模型推理  : {infer_time*1000:.1f}ms")
    print(f"  后处理    : {post_time*1000:.2f}ms")

    if trueLabel is not None:
        is_correct = trueLabel == Common.labels[pred]
        print(f"判断结果：{'✓ 正确' if is_correct else '✗ 错误'}")

    return total * 1000


# ============================================================
# 每类 1 张图片，推理 10 次（去除冷启动影响）
# ============================================================
def benchmark_per_class(modelPath, repeat=10):
    """对每个类别抽取 1 张图片，预热后重复推理 repeat 次，记录耗时统计"""
    from model import model as weatherModel

    run_dir, run_idx = get_latest_model_dir()
    SF = f"_model_{run_idx}"  # 文件名后缀，与文件夹编号匹配

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---------- 测量冷启动时间（模型加载 + 首次推理）----------
    t_cold_start = time.time()
    model = weatherModel
    model.load_state_dict(torch.load(modelPath, map_location=Common.device))
    model = model.to(Common.device)
    model.eval()
    if Common.device.type == 'cuda':
        torch.cuda.synchronize()
    load_time = time.time() - t_cold_start

    # 预热（让 cuDNN kernel 完成编译）
    dummy_path = None
    for label_name in Common.labels:
        label_dir = os.path.join(Common.basePath, label_name)
        if os.path.isdir(label_dir):
            files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            if files:
                dummy_path = os.path.join(label_dir, files[0])
                break
    dummy_img = Image.open(dummy_path).convert('RGB')
    dummy_tensor = transform(dummy_img).unsqueeze(0).to(Common.device)
    dummy_img.close()
    with torch.no_grad():
        _ = model(dummy_tensor)
    if Common.device.type == 'cuda':
        torch.cuda.synchronize()
    warmup_time = time.time() - t_cold_start - load_time

    cold_start_total = load_time + warmup_time
    print(f"模型加载耗时: {load_time*1000:.1f}ms")
    print(f"GPU 预热耗时: {warmup_time*1000:.1f}ms")
    print(f"冷启动总耗时: {cold_start_total*1000:.1f}ms")

    # 保存冷启动记录到单独文件
    cold_path = os.path.join(run_dir, f"cold_start{SF}.txt")
    with open(cold_path, 'w', encoding='utf-8') as f:
        f.write("冷启动耗时记录\n")
        f.write("=" * 40 + "\n")
        f.write(f"模型路径: {modelPath}\n")
        f.write(f"模型加载耗时: {load_time*1000:.2f} ms\n")
        f.write(f"GPU 预热耗时: {warmup_time*1000:.2f} ms\n")
        f.write(f"冷启动总耗时: {cold_start_total*1000:.2f} ms\n")
    print(f"冷启动记录已保存至: {cold_path}")

    # ---------- 每类 1 张图，重复推理 10 次 ----------
    results = []
    print("\n========== Per-Class Benchmark (after warmup, 10 runs) ==========")
    print(f"{'类别':<10} {'图片名':<30} {'最小(ms)':<10} {'最大(ms)':<10} {'平均(ms)':<10} {'std(ms)':<10}")
    print("-" * 80)

    for label_name in Common.labels:
        label_dir = os.path.join(Common.basePath, label_name)
        if not os.path.isdir(label_dir):
            continue
        image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        if not image_files:
            continue

        image_file = image_files[0]
        image_path = os.path.join(label_dir, image_file)

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(Common.device)
        image.close()

        times = []
        with torch.no_grad():
            for _ in range(repeat):
                t0 = time.time()
                _ = model(image_tensor)
                if Common.device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.time()
                times.append((t1 - t0) * 1000)

        times = np.array(times)
        results.append({
            "class": label_name,
            "image": image_file,
            "min": times.min(),
            "max": times.max(),
            "avg": times.mean(),
            "std": times.std()
        })
        print(f"{label_name:<10} {image_file:<30} {times.min():<10.2f} {times.max():<10.2f} {times.mean():<10.2f} {times.std():<10.2f}")

    print("-" * 80)

    all_avgs = [r["avg"] for r in results]
    print(f"{'总体':<10} {'-':<30} {min(all_avgs):<10.2f} {max(all_avgs):<10.2f} {np.mean(all_avgs):<10.2f} {np.std(all_avgs):<10.2f}")

    # 保存到 model_X 目录（不包含冷启动）
    table_path = os.path.join(run_dir, f"benchmark_per_class{SF}.csv")
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("类别,图片名,最小耗时(ms),最大耗时(ms),平均耗时(ms),标准差(ms)\n")
        for r in results:
            f.write(f"{r['class']},{r['image']},{r['min']:.2f},{r['max']:.2f},{r['avg']:.2f},{r['std']:.2f}\n")
        f.write(f"总体,-,{min(all_avgs):.2f},{max(all_avgs):.2f},{np.mean(all_avgs):.2f},{np.std(all_avgs):.2f}\n")

    print(f"\n基准测试表格已保存至: {table_path}")
    return results


if __name__ == '__main__':
    import sys

    run_dir, run_idx = get_latest_model_dir()
    SF = f"_model_{run_idx}"
    default_model_path = os.path.join(run_dir, f"best{SF}.pt")

    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_per_class(default_model_path, repeat=10)
    else:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = r"C:\Users\Lenovo\OneDrive\Desktop\caip_code\data\RSCM\classification\weather_classification\sunrise\sunrise2.jpg"
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
        else:
            model_path = default_model_path

        pridect(image_path, model_path, trueLabel="sunrise")