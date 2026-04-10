"""
GPU 模型預載測試腳本
測試所有 pipeline 模型同時載入 GPU 的可行性，記錄 VRAM 使用量與載入時間。
涵蓋：Aneurysm (Brain/Vessel/Aneurysm nnUNet + Vessel16 TF)、Infarct、WMH
"""

import os
import sys
import time
import argparse
import traceback

# 設定路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'nnResUNet_long_BigBatch_cosine_AneDilate_classifier_test'))


def get_all_gpu_info():
    """用 pynvml 列出所有 GPU 的 VRAM 使用量（物理 index）"""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        infos = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            infos.append({
                'index': i,
                'name': name,
                'used_mb': info.used / 1024 / 1024,
                'total_mb': info.total / 1024 / 1024,
            })
        pynvml.nvmlShutdown()
        return infos
    except Exception as e:
        print(f"  [WARN] pynvml 無法使用: {e}")
        return []


def get_torch_gpu_mem():
    """用 PyTorch API 取得 GPU 記憶體（精確追蹤 PyTorch 分配）"""
    import torch
    if not torch.cuda.is_available():
        return 0, 0, 0
    allocated = torch.cuda.memory_allocated(0) / 1024 / 1024  # 目前分配
    reserved = torch.cuda.memory_reserved(0) / 1024 / 1024    # 目前保留（含 cache）
    max_allocated = torch.cuda.max_memory_allocated(0) / 1024 / 1024
    return allocated, reserved, max_allocated


def load_nnunet_model(model_dir, folds, checkpoint_name, plans_json_name, loader_module):
    """載入 nnUNet 模型到 GPU (cuda:0，由 CUDA_VISIBLE_DEVICES 決定實際 GPU)"""
    import torch

    if loader_module == 'gpu_nnUNet':
        from gpu_nnUNet import load_what_we_need
    else:
        from gpu_nnUNet2D import load_what_we_need

    parameters, configuration_manager, inference_allowed_mirroring_axes, \
        plans_manager, dataset_json, network, trainer_name = \
        load_what_we_need(model_dir, folds, checkpoint_name, plans_json_name)

    # 載入第一組權重到 network 並搬到 GPU
    network.load_state_dict(parameters[0])
    device = torch.device('cuda:0')
    network = network.to(device)
    network.eval()

    return {
        'network': network,
        'parameters': parameters,
        'configuration_manager': configuration_manager,
        'plans_manager': plans_manager,
        'dataset_json': dataset_json,
        'trainer_name': trainer_name,
        'device': device,
    }


def load_tf_saved_model(model_path):
    """載入 TensorFlow SavedModel 到 GPU"""
    import tensorflow as tf
    model = tf.saved_model.load(model_path)
    model.trainable = False
    return model


def main():
    parser = argparse.ArgumentParser(description='GPU 模型預載測試')
    parser.add_argument('--gpu', type=int, default=0, help='物理 GPU index')
    parser.add_argument('--skip-tf', action='store_true', help='跳過 TensorFlow 模型')
    args = parser.parse_args()

    physical_gpu = args.gpu

    # 必須在 import torch/tf 之前設定
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # ============================================================
    # 模型路徑定義
    # ============================================================
    NNUNET_RESULTS = os.path.join(SCRIPT_DIR, 'nnUNet', 'nnUNet_results')
    MODEL_WEIGHTS = os.path.join(SCRIPT_DIR, 'model_weights')

    models_config = {
        # ---- Aneurysm Pipeline (PyTorch nnUNet 3D) ----
        'Aneurysm/Brain (nnUNet 3D)': {
            'type': 'nnunet',
            'loader': 'gpu_nnUNet',
            'model_dir': os.path.join(NNUNET_RESULTS, 'Dataset134_DeepMRABrain', 'nnUNetTrainer__nnUNetPlans__3d_fullres'),
            'folds': (0,),
            'checkpoint': 'checkpoint_best.pth',
            'plans_json': 'plans.json',
        },
        'Aneurysm/Vessel (nnUNet 3D)': {
            'type': 'nnunet',
            'loader': 'gpu_nnUNet',
            'model_dir': os.path.join(NNUNET_RESULTS, 'Dataset135_DeepMRAVessel', 'nnUNetTrainer__nnUNetPlans__3d_fullres'),
            'folds': (0,),
            'checkpoint': 'checkpoint_best.pth',
            'plans_json': 'plans.json',
        },
        'Aneurysm/Detection (nnUNet 3D)': {
            'type': 'nnunet',
            'loader': 'gpu_nnUNet',
            'model_dir': os.path.join(NNUNET_RESULTS, 'Dataset080_DeepAneurysm', 'nnUNetTrainer__nnUNetPlans__3d_fullres'),
            'folds': (5,),
            'checkpoint': 'checkpoint_best.pth',
            'plans_json': 'nnUNetPlans_5L-b900.json',
        },

        # ---- Infarct Pipeline (PyTorch nnUNet 2D) ----
        'Infarct (nnUNet 2D)': {
            'type': 'nnunet',
            'loader': 'gpu_nnUNet2D',
            'model_dir': os.path.join(NNUNET_RESULTS, 'Dataset040_DeepInfarct', 'nnUNetTrainer__nnUNetPlans__2d'),
            'folds': (0,),
            'checkpoint': 'checkpoint_best.pth',
            'plans_json': 'plans.json',
        },

        # ---- WMH Pipeline (PyTorch nnUNet 2D) ----
        'WMH/Lacune (nnUNet 2D)': {
            'type': 'nnunet',
            'loader': 'gpu_nnUNet2D',
            'model_dir': os.path.join(NNUNET_RESULTS, 'Dataset015_DeepLacune', 'nnUNetTrainer__nnUNetPlans__2d'),
            'folds': (1,),
            'checkpoint': 'checkpoint_final.pth',
            'plans_json': 'plans.json',
        },

        # ---- TensorFlow Models ----
        'Aneurysm/Vessel16 (TF SavedModel)': {
            'type': 'tf',
            'model_path': os.path.join(
                MODEL_WEIGHTS,
                'dataV2-Vessel_16label-ResUnet',
                'MP-Vessel_16label+branch_PostProc2AIAA_Aug2of4_160_b4-ResUnet_8f_L5_BN_mish_19ch_ema_cw',
                'best_saved_model'
            ),
        },
        'SynthSeg (TF)': {
            'type': 'tf',
            'model_path': os.path.join(MODEL_WEIGHTS, 'SynthSeg_parcellation_tf28'),
            'note': 'SynthSeg 載入方式較特殊，此處僅測試 TF GPU 是否可用',
        },
    }

    # ============================================================
    # GPU 資訊
    # ============================================================
    print("=" * 70)
    print("GPU 模型預載測試")
    print("=" * 70)

    # 列出所有 GPU（pynvml 用物理 index）
    print("\n[所有 GPU - pynvml 物理 index]")
    all_gpus = get_all_gpu_info()
    for g in all_gpus:
        marker = " <<< TARGET" if g['index'] == physical_gpu else ""
        print(f"  GPU {g['index']}: {g['name']} - {g['used_mb']:.0f} MB / {g['total_mb']:.0f} MB{marker}")

    print(f"\nCUDA_VISIBLE_DEVICES={physical_gpu} (PyTorch/TF 會看到 cuda:0)")

    # 初始化 PyTorch 並確認裝置
    import torch
    print(f"\nPyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  torch cuda:0 → {torch.cuda.get_device_name(0)}")
        # 觸發 CUDA context 初始化
        torch.zeros(1, device='cuda:0')
        torch.cuda.reset_peak_memory_stats(0)

    # Baseline
    alloc_base, reserved_base, _ = get_torch_gpu_mem()
    print(f"\n[Baseline] PyTorch allocated: {alloc_base:.0f} MB, reserved: {reserved_base:.0f} MB")

    loaded_models = {}
    results = []  # (name, status, elapsed, alloc_mb, delta_mb)

    # ============================================================
    # Phase 1: PyTorch nnUNet 模型
    # ============================================================
    print("\n" + "-" * 70)
    print("Phase 1: PyTorch nnUNet 模型載入")
    print("-" * 70)

    prev_alloc = alloc_base

    for name, cfg in models_config.items():
        if cfg['type'] != 'nnunet':
            continue

        print(f"\n  Loading: {name}")
        print(f"    Path: {cfg['model_dir']}")

        if not os.path.isdir(cfg['model_dir']):
            print(f"    [SKIP] 模型目錄不存在")
            results.append((name, 'SKIP', 0, 0, 0))
            continue

        t0 = time.time()
        try:
            model_data = load_nnunet_model(
                cfg['model_dir'], cfg['folds'], cfg['checkpoint'],
                cfg['plans_json'], cfg['loader']
            )
            elapsed = time.time() - t0
            alloc_now, reserved_now, _ = get_torch_gpu_mem()
            delta = alloc_now - prev_alloc

            loaded_models[name] = model_data
            results.append((name, 'OK', elapsed, alloc_now, delta))

            print(f"    Status: OK")
            print(f"    Time: {elapsed:.1f}s")
            print(f"    PyTorch allocated: {alloc_now:.0f} MB (+{delta:.0f} MB)")
            prev_alloc = alloc_now

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    Status: FAILED")
            print(f"    Error: {e}")
            traceback.print_exc()
            results.append((name, 'FAILED', elapsed, 0, 0))

    # PyTorch 小計
    alloc_after_pt, reserved_after_pt, peak_pt = get_torch_gpu_mem()
    print(f"\n  [PyTorch 小計] allocated: {alloc_after_pt:.0f} MB, reserved: {reserved_after_pt:.0f} MB, peak: {peak_pt:.0f} MB")

    # ============================================================
    # Phase 2: TensorFlow 模型
    # ============================================================
    if not args.skip_tf:
        print("\n" + "-" * 70)
        print("Phase 2: TensorFlow 模型載入")
        print("-" * 70)

        try:
            import tensorflow as tf
            print(f"TensorFlow {tf.__version__}")

            gpus = tf.config.list_physical_devices('GPU')
            print(f"TF visible GPUs: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"TensorFlow 初始化失敗: {e}")

        for name, cfg in models_config.items():
            if cfg['type'] != 'tf':
                continue

            print(f"\n  Loading: {name}")
            print(f"    Path: {cfg.get('model_path', 'N/A')}")

            if 'SynthSeg' in name:
                print(f"    [SKIP] SynthSeg 需要特殊載入方式")
                results.append((name, 'SKIP', 0, 0, 0))
                continue

            if not os.path.isdir(cfg.get('model_path', '')):
                print(f"    [SKIP] 模型目錄不存在")
                results.append((name, 'SKIP', 0, 0, 0))
                continue

            # TF 前用 pynvml 測 (因為 torch API 追蹤不到 TF 的分配)
            gpus_before = get_all_gpu_info()
            vram_before = gpus_before[physical_gpu]['used_mb'] if physical_gpu < len(gpus_before) else 0

            t0 = time.time()
            try:
                model = load_tf_saved_model(cfg['model_path'])
                elapsed = time.time() - t0

                gpus_after = get_all_gpu_info()
                vram_after = gpus_after[physical_gpu]['used_mb'] if physical_gpu < len(gpus_after) else 0
                delta = vram_after - vram_before

                loaded_models[name] = model
                results.append((name, 'OK', elapsed, vram_after, delta))

                print(f"    Status: OK")
                print(f"    Time: {elapsed:.1f}s")
                print(f"    GPU VRAM (pynvml): {vram_after:.0f} MB (+{delta:.0f} MB)")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"    Status: FAILED")
                print(f"    Error: {e}")
                traceback.print_exc()
                results.append((name, 'FAILED', elapsed, 0, 0))
    else:
        print("\n[SKIP] TensorFlow 模型（--skip-tf）")

    # ============================================================
    # 結果摘要
    # ============================================================
    print("\n" + "=" * 70)
    print("測試結果摘要")
    print("=" * 70)

    # 最終 PyTorch 記憶體
    final_alloc, final_reserved, final_peak = get_torch_gpu_mem()

    # 最終 pynvml (所有 GPU)
    final_gpus = get_all_gpu_info()
    target_gpu = final_gpus[physical_gpu] if physical_gpu < len(final_gpus) else None

    print(f"\n{'模型名稱':<35} {'狀態':<8} {'載入時間':<10} {'增量 MB':<10}")
    print("-" * 70)
    for name, status, elapsed, mem, delta in results:
        delta_str = f"+{delta:.0f}" if delta > 0 else f"{delta:.0f}" if status == 'OK' else "-"
        print(f"{name:<35} {status:<8} {elapsed:>6.1f}s    {delta_str:<10}")

    print("-" * 70)
    print(f"\n[PyTorch GPU 記憶體]")
    print(f"  Allocated: {final_alloc:.0f} MB")
    print(f"  Reserved:  {final_reserved:.0f} MB")
    print(f"  Peak:      {final_peak:.0f} MB")

    if target_gpu:
        print(f"\n[GPU {physical_gpu} 實際 VRAM (pynvml)]")
        print(f"  {target_gpu['name']}")
        print(f"  Used:  {target_gpu['used_mb']:.0f} MB / {target_gpu['total_mb']:.0f} MB "
              f"({target_gpu['used_mb']/target_gpu['total_mb']*100:.1f}%)")
        remaining = target_gpu['total_mb'] - target_gpu['used_mb']
        print(f"  Free:  {remaining:.0f} MB")

    ok_count = sum(1 for _, s, _, _, _ in results if s == 'OK')
    fail_count = sum(1 for _, s, _, _, _ in results if s == 'FAILED')
    total_count = sum(1 for _, s, _, _, _ in results if s != 'SKIP')
    print(f"\n成功: {ok_count}/{total_count}, 失敗: {fail_count}/{total_count}")

    if target_gpu:
        remaining = target_gpu['total_mb'] - target_gpu['used_mb']
        if remaining > 2000:
            print("結論: VRAM 充足，可支援所有模型常駐預載")
        elif remaining > 500:
            print("結論: VRAM 偏緊，建議分批載入或使用更大 VRAM 的 GPU")
        else:
            print("結論: VRAM 不足，不建議全部模型同時常駐")

    # 等待確認
    print("\n所有模型已載入 GPU 記憶體中。")
    print("按 Enter 釋放所有模型並結束...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    # 清理
    print("正在釋放模型...")
    loaded_models.clear()
    torch.cuda.empty_cache()
    if 'tensorflow' in sys.modules:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except:
            pass

    after_gpus = get_all_gpu_info()
    if physical_gpu < len(after_gpus):
        released = target_gpu['used_mb'] - after_gpus[physical_gpu]['used_mb']
        print(f"釋放後 GPU {physical_gpu}: {after_gpus[physical_gpu]['used_mb']:.0f} MB (釋放了 {released:.0f} MB)")
    print("完成。")


if __name__ == '__main__':
    main()
