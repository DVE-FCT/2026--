"""
SupCon λ × τ 网格搜索（支持中断继续）
自动遍历所有组合，训练 → 测试 → 记录结果
"""
import os, sys, time, itertools, importlib, random
import pandas as pd

sys.path.insert(0, ".")
import config as _cfg
from config import Train, Common, SEED

GRID_LAMBDAS = [0.03, 0.04, 0.05, 0.07, 0.1]
GRID_TAUS    = [0.07, 0.1, 0.15, 0.2, 0.3]
CSV_PATH = "./model/supcon_grid_search.csv"


def reset_run_seed():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)


def load_completed():
    """读取已完成的结果，返回 {(λ, τ)} 集合"""
    if not os.path.exists(CSV_PATH):
        return set()
    df = pd.read_csv(CSV_PATH)
    completed = set()
    for _, row in df.iterrows():
        if row.get("Model", "ERR") != "ERR":
            completed.add((row["lambda"], row["tau"]))
    return completed


def save_results(results):
    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    return df


def run_one(lambda_val, tau_val):
    """训练 + 测试一个 (λ, τ) 组合"""
    reset_run_seed()
    Train.supcon_enabled = True
    Train.supcon_lambda = lambda_val
    Train.supcon_temperature = tau_val
    Train.num_workers = 2

    import model as _m
    importlib.reload(_m)
    import data_loader as _dl
    importlib.reload(_dl)
    import train as _t
    importlib.reload(_t)
    import test as _test
    importlib.reload(_test)

    print(f"\n--- 训练参数 λ={lambda_val}, τ={tau_val} ---")
    _t.main()

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)  # 确保 worker 完全退出

    run_dir, run_idx = _test.get_latest_model_dir()
    sf = f"_model_{run_idx}"
    model_path = os.path.join(run_dir, f"best{sf}.pt")

    _test.run_dir = run_dir
    _test.run_idx = run_idx
    _test.SF = sf
    _test.model_path = model_path
    _test.test()

    result_path = os.path.join(run_dir, f"test_result{sf}.txt")
    acc = f1 = 0.0
    with open(result_path, encoding='utf-8') as fh:
        for line in fh:
            if 'Test Acc' in line:
                acc = float(line.split(':')[1].strip().split()[0])
            if 'Macro F1' in line:
                f1 = float(line.split()[-1])
    # 读取最佳 epoch
    log_path = os.path.join(run_dir, f"training_log{sf}.txt")
    best_ep = 0
    with open(log_path, encoding='utf-8') as fh:
        for line in fh:
            if '最佳 epoch' in line:
                best_ep = int(line.split(':')[1].strip().split('/')[0])
                break
    return run_idx, acc, f1, best_ep


def main():
    completed = load_completed()
    grid_all = list(itertools.product(GRID_LAMBDAS, GRID_TAUS))
    grid_todo = [(l, t) for l, t in grid_all if (l, t) not in completed]

    if completed:
        print(f"检测到已完成 {len(completed)}/{len(grid_all)} 组，跳过")
        for l, t in sorted(completed):
            print(f"  ✓ λ={l}, τ={t}")
    print(f"剩余 {len(grid_todo)}/{len(grid_all)} 组待运行")

    if not grid_todo:
        print("全部完成！")
        df = pd.read_csv(CSV_PATH)
        print(df.to_string(index=False))
        return

    # 加载已有结果
    results = []
    if os.path.exists(CSV_PATH):
        results = pd.read_csv(CSV_PATH).to_dict('records')

    pbar = __import__('tqdm').tqdm(grid_todo, desc="网格搜索 (λ×τ)", ncols=100, unit="组",
                                   initial=len(completed), total=len(grid_all))
    for lam, tau in pbar:
        pbar.set_postfix({"λ": lam, "τ": tau, "状态": "训练中"})
        try:
            idx, acc, f1, best_ep = run_one(lam, tau)
            results.append({"Model": f"M{idx}", "lambda": lam, "tau": tau,
                            "Acc": acc, "Macro_F1": f1, "Best_Epoch": best_ep})
            save_results(results)  # 每组跑完立刻存
            pbar.set_postfix({"λ": lam, "τ": tau, "状态": f"M{idx} Acc={acc:.4f}"})
        except Exception:
            import traceback
            traceback.print_exc()
            results.append({"Model": "ERR", "lambda": lam, "tau": tau,
                            "Acc": 0, "Macro_F1": 0, "Best_Epoch": 0})
            save_results(results)
            pbar.set_postfix({"λ": lam, "τ": tau, "状态": "失败"})

    df = save_results(results)
    print(f"\n{'='*60}")
    print("网格搜索完成")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\n基线 M40（无 SupCon）请参考已有结果")
    print(f"结果已保存: {CSV_PATH}")


if __name__ == '__main__':
    main()
