"""
天气识别项目一键式运行脚本
功能：训练 → 测试 → 推理分析（每类 benchmark 10 次）→ T-SNE 可视化
用法：python main.py [阶段]
阶段可选：train / test / infer / tsne / all（默认 all）
"""

import sys
import subprocess
import os

# 当前项目目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def run_script(script_name, description, args=None):
    """运行单个脚本并打印分隔线"""
    print(f"\n{'='*60}")
    print(f" 阶段：{description}")
    print(f" 脚本：{script_name}")
    print(f"{'='*60}\n")
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"[错误] {script_name} 执行失败，退出码: {result.returncode}")
        sys.exit(1)
    print(f"[完成] {description}")


def main():
    if len(sys.argv) < 2:
        stage = "all"
    else:
        stage = sys.argv[1].lower()

    print(f"\n{'#'*60}")
    print(f"# 天气识别 - 一键式运行")
    print(f"# 阶段: {stage}")
    print(f"# 工作目录: {PROJECT_DIR}")
    print(f"{'#'*60}")

    if stage in ("train", "all"):
        run_script("train.py", "模型训练")

    if stage in ("test", "all"):
        run_script("test.py", "模型测试")

    if stage in ("infer", "all"):
        # 推理分析 = 运行 pridect.py --benchmark（每类图片推理10次取平均）
        run_script("pridect.py", "推理分析（Benchmark + 单张推理演示）", args=["--benchmark"])

    if stage in ("tsne", "all"):
        run_script("tsne_visualization.py", "T-SNE 降维可视化")

    print(f"\n{'#'*60}")
    print(f"# 全部完成！")
    print(f"# 输出目录: ./model/model_X/（X 为最新模型编号）")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()