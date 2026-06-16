"""
模型验证脚本：M40 基线 vs 粗细双头模型，每 ep 更新 alpha，仅 10 epoch
"""
import os, sys, time, importlib
sys.path.insert(0, ".")
from config import Train, Common, SEED

EXPERIMENTS = [
    # 参数: (Label, dual_head, alpha_power, use_consistency, coarse_lambda)
    #   dual_head: True=粗(2类)+细(8类)双头, False=仅8类
    #   alpha_power: difficulty^pow, 0.5=压缩差异, 1.0=原M22
    #   use_consistency: KL(P_fine_merged || P_coarse_head), fine→coarse寻对齐
    #   coarse_lambda: 粗分类CE权重 (total=focal+λ×CE_coarse)
    # pow 消融 (已完成)
    ("M40 baseline",                  False,  0.5,  False, 0.0),
    ("M40 + pow=0.75",                False,  0.75, False, 0.0),
    ("M40 + pow=1.0",                 False,  1.0,  False, 0.0),
    # 双头消融 (已完成) — pow=0.5 固定
    ("Dual λ=0.1 (no KL)",            True,   0.5,  False, 0.1),
    ("Dual λ=0.2 (no KL)",            True,   0.5,  False, 0.2),
    ("Dual λ=0.1 + KL λ=0.01",        True,   0.5,  True,  0.1),
    # 专家消融 — pow=0.5 + coarse λ=0.1 固定 (待添加)
]


def reset_seed():
    import random, numpy as np, torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def run_one(label, use_dual_head, alpha_power, use_consistency, coarse_lambda):
    reset_seed()
    Train.backbone = "resnet50"
    Train.use_se_attention = False
    Train.supcon_enabled = False
    Train.dynamic_alpha_interval = 1
    Train.dynamic_alpha_warmup = 0
    Train.dynamic_alpha_learned = False
    Train.alpha_power = alpha_power
    Train.use_coarse_head = use_dual_head
    Train.coarse_lambda = coarse_lambda
    Train.use_consistency_loss = use_consistency
    Train.consistency_lambda = 0.01 if use_consistency else 0
    Train.consistency_start_epoch = 5 if use_consistency else 0
    Train.use_expert_head = True if use_dual_head else getattr(Train, 'use_expert_head', False)
    Train.expert_lambda = 0.2 if use_dual_head else 0
    Train.batch_size = 128
    Train.num_workers = 2  # 连续跑降 worker 防 DLL 冲突
    Train.epochs = 10
    Train.early_stop_enabled = False

    for mod in ['model', 'data_loader', 'train', 'test']:
        sys.modules.pop(mod, None)

    import model as _m
    import data_loader as _dl
    import train as _t
    import test as _test

    print(f"\n{'='*60}")
    print(f"验证: {label} (power={alpha_power}, dual_head={use_dual_head})")
    print(f"{'='*60}")
    _t.main()

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(8)  # 确保上轮 worker 完全退出

    run_dir, run_idx = _test.get_latest_model_dir()
    sf = f"_model_{run_idx}"
    _test.run_dir = run_dir
    _test.run_idx = run_idx
    _test.SF = sf
    _test.model_path = os.path.join(run_dir, f"best{sf}.pt")
    _test.test()

    result_path = os.path.join(run_dir, f"test_result{sf}.txt")
    acc = f1 = 0.0
    with open(result_path, encoding='utf-8') as fh:
        for line in fh:
            if 'Test Acc' in line:
                acc = float(line.split(':')[1].strip().split()[0])
            if 'Macro F1' in line:
                f1 = float(line.split()[-1])
    return run_idx, acc, f1


def main():
    import pandas as pd
    csv_path = "./model/validate_results.csv"
    completed = set()
    results = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        results = df.to_dict('records')
        for _, row in df.iterrows():
            if row.get("Model", "ERR") != "ERR":
                completed.add(row["Label"])
    todo = [(l, d, p, c, cl) for l, d, p, c, cl in EXPERIMENTS if l not in completed]
    if not todo:
        print("全部完成！")
        print(pd.read_csv(csv_path).to_string(index=False))
        return
    if completed:
        print(f"已完成 {len(completed)}/{len(EXPERIMENTS)} 组")

    pbar = __import__('tqdm').tqdm(todo, desc="验证", ncols=100, unit="组")
    for label, dual_head, power, consistency, lam in pbar:
        pbar.set_postfix({"标签": label, "状态": "训练中"})
        try:
            idx, acc, f1 = run_one(label, dual_head, power, consistency, lam)
            results.append({"Label": label, "Model": f"M{idx}", "Acc": acc, "Macro_F1": f1})
            pd.DataFrame(results).to_csv(csv_path, index=False)
            pbar.set_postfix({"标签": label, "状态": f"M{idx} F1={f1:.4f}"})
        except Exception:
            import traceback
            traceback.print_exc()
            results.append({"Label": label, "Model": "ERR", "Acc": 0, "Macro_F1": 0})
            pd.DataFrame(results).to_csv(csv_path, index=False)
            pbar.set_postfix({"标签": label, "状态": "失败"})

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\n结果: {csv_path}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
