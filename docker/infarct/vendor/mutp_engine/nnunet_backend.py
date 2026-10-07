"""NnUNetBackend — 將 RecipeConfig 完整對映到 nnU-Net run_training 的所有參數。

不用 subprocess 呼叫 nnUNetv2_train，而是直接 import run_training()，
這樣可以：
1. 在訓練前後加入 MUTP 的 hook（MLflow、自動推論、FROC 計算）
2. 覆寫 Trainer class（MutpNnUNetTrainer）
3. 控制所有參數不遺漏

來源：nnResUNet-location-cls-upsample/nnunetv2/run/run_training.py
"""

import json
import os
import pathlib
import subprocess
import sys
from typing import Optional

import torch

from mutp.config.recipe_schema import RecipeConfig


import threading
import time


class _MLflowWatcherThread(threading.Thread):
    """背景執行緒：監控 training log，每個 epoch 完成即同步到 MLflow。"""

    def __init__(self, recipe, experiment_dir, poll_interval=30):
        super().__init__(daemon=True)
        self.recipe = recipe
        self.experiment_dir = experiment_dir
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._synced_epochs = 0
        self._tracker = None

    def stop(self):
        """通知停止，並做最終同步。"""
        self._stop_event.set()

    def run(self):
        try:
            from mutp.cli.main import _parse_training_log, _find_fold_dir
            from mutp.tracking.mlflow_tracker import MutpMLflowTracker

            # 等 fold_dir 出現（訓練可能還沒開始寫 log）
            fold_dir = None
            for _ in range(60):  # 最多等 10 分鐘
                fold_dir, _, _ = _find_fold_dir(self.experiment_dir)
                if fold_dir:
                    break
                if self._stop_event.wait(10):
                    return
            if fold_dir is None:
                return

            # 啟動 MLflow run
            self._tracker = MutpMLflowTracker(self.recipe)
            run_id = self._tracker.start_run()
            print(f"[MUTP MLflow] 背景同步啟動 (run_id={run_id[:8]}...)")

            while not self._stop_event.is_set():
                self._sync_new_epochs(fold_dir)
                self._stop_event.wait(self.poll_interval)

            # 最終同步
            self._sync_new_epochs(fold_dir)

            # 記錄 progress.png
            progress_png = fold_dir / "progress.png"
            if progress_png.exists():
                import mlflow
                mlflow.log_artifact(str(progress_png), "training_curves")

            self._tracker.end_run()
            print(f"[MUTP MLflow] 背景同步結束（共 {self._synced_epochs} epochs）")
        except Exception as err:
            print(f"[MUTP MLflow] 背景同步異常（非致命）：{err}")

    def _sync_new_epochs(self, fold_dir):
        """解析 log，同步尚未寫入的 epoch。"""
        from mutp.cli.main import _parse_training_log

        log_files = sorted(fold_dir.glob("training_log_*.txt"), key=lambda f: f.stat().st_mtime)
        if not log_files:
            return

        epochs = _parse_training_log(log_files[-1])
        new_epochs = epochs[self._synced_epochs:]
        if not new_epochs:
            return

        for ep in new_epochs:
            metrics = {}
            if "train_loss" in ep:
                metrics["train_loss"] = ep["train_loss"]
            if "val_loss" in ep:
                metrics["val_loss"] = ep["val_loss"]
            if "train_dice" in ep:
                for i, d in enumerate(ep["train_dice"]):
                    metrics[f"train_dice_ch{i}"] = d
            if "val_dice" in ep:
                for i, d in enumerate(ep["val_dice"]):
                    metrics[f"val_dice_ch{i}"] = d
            if "lr" in ep:
                try:
                    metrics["lr"] = float(ep["lr"])
                except ValueError:
                    pass
            self._tracker.log_epoch(ep["epoch"], metrics)

        self._synced_epochs = len(epochs)


def _start_mlflow_watcher(recipe, experiment_dir, poll_interval=30):
    """啟動 MLflow 背景同步執行緒。回傳 thread 物件。"""
    watcher = _MLflowWatcherThread(recipe, experiment_dir, poll_interval)
    watcher.start()
    return watcher


def _parse_comma_str_to_list(s: Optional[str], dtype=float) -> Optional[list]:
    """'5,1,1,1,1' → [5.0, 1.0, 1.0, 1.0, 1.0]"""
    if s is None:
        return None
    return [dtype(x.strip()) for x in s.split(",")]


def _parse_sampling_weights(s: Optional[str]) -> Optional[dict]:
    """支援任意數量的 category weights。

    格式：
    - '1:1' → {1: 1.0, 2: 1.0}（2 類）
    - '2:1:1:1' → {1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0}（4 類）
    - '1=2,2=1' → {1: 2.0, 2: 1.0}（key=value）
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # key=value 格式
    if "=" in s:
        result = {}
        for part in s.replace(";", ",").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                result[int(k.strip())] = float(v.strip())
        return result
    # ratio 格式：自動 map 到 1..N
    parts = [p.strip() for p in s.replace(":", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    return {i + 1: v for i, v in enumerate(vals)}


def _parse_normal_weights(s: Optional[str]) -> Optional[dict]:
    """'1=2,2=1,3=1,4=1' → {1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0}"""
    if s is None:
        return None
    result = {}
    for part in s.replace(";", ",").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            result[int(k.strip())] = float(v.strip())
    return result


def _parse_int_list(s: Optional[str]) -> Optional[list]:
    """'1,2' → [1, 2]"""
    if s is None:
        return None
    return [int(x.strip()) for x in s.split(",")]


def _cleanup_pymp_orphans(min_age_seconds: int = 600) -> int:
    """Clean orphan /tmp/pymp-* dirs (Python multiprocessing named-object dirs).

    These accumulate when Python multiprocessing pools/queues die without
    cleanup (e.g. process killed, container restarted, worker crash).
    When thousands accumulate, new multiprocessing.spawn calls hang because
    the resource_tracker takes forever to enumerate them.

    Safety (兩層防護，避免砍到正在跑的 training 的 pymp-*)：
      1. 掃 /proc/net/unix 找 pymp-*/listener-* Unix socket path → 這些絕不砍
         （pymp dir 裡的 listener 是 Unix socket，不是普通檔案，/proc/*/fd/* readlink
         會回 socket:[N] 抓不到 path。/proc/net/unix 才有 socket path 清單）
         也順便掃 /proc/*/fd/* 抓 regular file handle，double coverage
      2. 剩下的候選再看 mtime，`min_age_seconds` 以下的 skip
         （保險 fallback：若掃描漏抓某些 dir，剛建的先 skip）

    Returns number of dirs removed.
    """
    import time as _time
    import glob as _glob
    import shutil as _shutil

    # Step 1a: /proc/net/unix — Unix socket path (pymp listener 都在這)
    in_use = set()
    try:
        with open("/proc/net/unix") as _f:
            for _line in _f:
                _idx = _line.find("/tmp/pymp-")
                if _idx == -1:
                    continue
                _path = _line[_idx:].strip()
                _slash = _path.find("/", len("/tmp/pymp-"))
                in_use.add(_path if _slash == -1 else _path[:_slash])
    except OSError:
        pass

    # Step 1b: /proc/*/fd/* — regular file handle（雙保險）
    try:
        for fd_link in _glob.glob("/proc/*/fd/*"):
            try:
                target = os.readlink(fd_link)
                if target.startswith("/tmp/pymp-"):
                    slash = target.find("/", len("/tmp/pymp-"))
                    in_use.add(target if slash == -1 else target[:slash])
            except (OSError, FileNotFoundError):
                continue
    except OSError:
        pass

    now = _time.time()
    removed = 0
    orphan_candidates = _glob.glob("/tmp/pymp-*")
    for d in orphan_candidates:
        try:
            if d in in_use:
                continue  # 有 process 拿著 fd，絕對不能砍
            mtime = os.path.getmtime(d)
            if now - mtime < min_age_seconds:
                continue  # 剛建的先 skip（保險）
            _shutil.rmtree(d, ignore_errors=True)
            if not os.path.exists(d):
                removed += 1
        except OSError:
            continue
    if removed > 0:
        print(f"[MUTP] cleaned {removed}/{len(orphan_candidates)} orphan /tmp/pymp-* dirs "
              f"(no active fd, >{min_age_seconds}s old; {len(in_use)} in-use dirs skipped)")
    return removed


def _find_aux_seg_location(exp_path, source_dir: Optional[str], configuration: str):
    """Locate aux seg files. Three layouts checked in order (newest first):

      (A) inline:        <root>/nnUNet_preprocessed/Dataset*/nnUNetPlans_<config>/{case}_aux_seg.npz
                         (preferred — alongside other preprocessed case files)
      (B) nested folder: <root>/nnUNet_preprocessed/Dataset*/nnUNetPlans_<config>/aux_seg/{case}.npz
                         (intermediate layout, kept for back-compat)
      (C) legacy top:    <root>/aux_seg/{case}.npz
                         (oldest layout)

    Returns: ("inline", <preprocessed_data_dir>) | ("folder", <aux_seg_dir>) | None
    """
    import pathlib as _pl
    roots = []
    if exp_path:
        roots.append(_pl.Path(exp_path))
    if source_dir:
        roots.append(_pl.Path(source_dir))
    for root in roots:
        for ds_dir in (root / "nnUNet_preprocessed").glob("Dataset*"):
            data_dir = ds_dir / f"nnUNetPlans_{configuration}"
            # (A) inline files in data_dir
            if data_dir.is_dir() and any(data_dir.glob("*_aux_seg.npz")):
                return ("inline", data_dir)
            # (B) nested aux_seg/ folder
            nested = data_dir / "aux_seg"
            if nested.is_dir() and any(nested.iterdir()):
                return ("folder", nested)
        # (C) legacy top-level
        legacy = root / "aux_seg"
        if legacy.is_dir() and any(legacy.iterdir()):
            return ("folder", legacy)
    return None


def _build_loss_config(recipe) -> Optional[dict]:
    """從 recipe.loss.components 抽出所有 loss component，組成 trainer LOSS_CONFIG。

    支援單 loss 跟複合 loss：
      - 單一：components: [{name: 'Tversky_and_CE_loss', weight: 1, params: {alpha: 0.3}}]
      - 複合：components: [
            {name: 'Tversky_and_CE_loss', weight: 1, params: {alpha: 0.3, beta: 0.7}},
            {name: 'BoundaryLoss', weight: 0.5, params: {idc: [1], normalize: true}},
          ]
    Trainer 端會根據 components 數量自動決定 single 或 Compound_loss wrapper。

    回傳 None = 不傳 LOSS_CONFIG → trainer 用預設 DC_and_CE_loss。
    """
    if not (hasattr(recipe, "loss") and recipe.loss and recipe.loss.components):
        return None

    components = []
    for comp in recipe.loss.components:
        name = getattr(comp, "name", None)
        if name in (None, ""):
            continue
        components.append({
            "name": name,
            "weight": float(getattr(comp, "weight", 1.0)),
            "params": dict(getattr(comp, "params", {}) or {}),
        })

    if not components:
        return None

    # 若只有一個 DC_and_CE_loss + 預設 weight + 無 params → 沿用預設、不傳
    if (len(components) == 1
            and components[0]["name"] == "DC_and_CE_loss"
            and components[0]["weight"] == 1.0
            and not components[0]["params"]):
        return None

    return {"components": components}


class NnUNetBackend:
    """nnU-Net 訓練後端 — 完整對映 RecipeConfig → run_training() 參數。"""

    def __init__(self, recipe: RecipeConfig):
        self.recipe = recipe
        self._setup_env_vars()

    def _setup_env_vars(self):
        """設定 nnU-Net 環境變數。優先使用 experiment_dir，否則用全局環境變數。"""
        exp_dir = self.recipe.engine.experiment_dir
        if exp_dir:
            exp_path = pathlib.Path(exp_dir)
            self._nnunet_env = {
                "nnUNet_raw": str(exp_path / "nnUNet_raw"),
                "nnUNet_preprocessed": str(exp_path / "nnUNet_preprocessed"),
                "nnUNet_results": str(exp_path / "nnUNet_results"),
            }
            for d in self._nnunet_env.values():
                pathlib.Path(d).mkdir(parents=True, exist_ok=True)
            # 設定到 os.environ（供 import 的 nnU-Net 模組使用）
            for k, v in self._nnunet_env.items():
                os.environ[k] = v
            print(f"[MUTP] experiment_dir: {exp_dir}")
            print(f"  nnUNet_raw: {self._nnunet_env['nnUNet_raw']}")
            print(f"  nnUNet_preprocessed: {self._nnunet_env['nnUNet_preprocessed']}")
            print(f"  nnUNet_results: {self._nnunet_env['nnUNet_results']}")
        else:
            # 用全局環境變數
            self._nnunet_env = {}
            required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
            for var in required:
                if var not in os.environ:
                    raise EnvironmentError(
                        f"環境變數 {var} 未設定，且 recipe 未指定 engine.experiment_dir。\n"
                        f"請設定環境變數或在 recipe 中加入 engine.experiment_dir。"
                    )

    def run(self) -> dict:
        """執行訓練。回傳結果 dict。"""
        _cleanup_pymp_orphans()   # 清 Python multiprocessing 遺留的孤兒目錄（>10 min 未活動的）

        r = self.recipe
        t = r.training
        e = r.engine
        m = r.model
        o = r.optimizer
        s = r.scheduler
        es = t.early_stopping
        loss = r.loss

        # 解析字串格式參數（只有 enabled 時才解析值，否則傳 None 讓 nnU-Net 知道是關閉）
        sampling_weights = _parse_sampling_weights(t.sampling_category_weights) if t.enable_sampling_weights else None
        normal_weights = _parse_normal_weights(t.normal_class_weights) if t.enable_normal_upsample else None
        region_weights = _parse_comma_str_to_list(loss.region_loss_weights) if loss.enable_region_loss_weights else None
        cls_fg_labels = _parse_int_list(m.classifier_head.foreground_labels) if m.classifier_head.enabled else None
        best_val_cls = _parse_int_list(t.best_val_classes) if t.enable_best_val_classes else None

        # 決定 dataset ID（自動偵測）
        dataset_id = r.data.nnunet_dataset_id
        if dataset_id is None and e.experiment_dir:
            import pathlib
            raw_dir = pathlib.Path(e.experiment_dir) / "nnUNet_raw"
            if raw_dir.exists():
                ds_dirs = sorted(raw_dir.glob("Dataset*_*"))
                if ds_dirs:
                    try:
                        dataset_id = int(ds_dirs[0].name.split("_")[0].replace("Dataset", ""))
                        print(f"[MUTP] 自動偵測 dataset_id={dataset_id}")
                    except ValueError:
                        pass
        if dataset_id is None:
            raise ValueError("需要 data.nnunet_dataset_id（recipe 未指定且無法從 experiment_dir 偵測）")

        configuration = r.data.nnunet_configuration or "3d_fullres"

        # 決定 trainer class
        trainer_class = e.nnunet_trainer_class or "nnUNetTrainer"

        plans = e.nnunet_plans or "nnUNetPlans"

        # Device — 從 recipe 的 device 設定 CUDA_VISIBLE_DEVICES
        # num_gpus>1 時 DDP 要 expose 多顆 GPU，從 device 指定的 index 開始連續 N 顆
        import os
        device_str = t.device if t.device else "cuda:0"
        if ":" in device_str:
            gpu_id = device_str.split(":")[1]
            if t.num_gpus and t.num_gpus > 1:
                start = int(gpu_id)
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(start + i) for i in range(t.num_gpus))
                print(f"[MUTP] DDP num_gpus={t.num_gpus}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            device = torch.device("cuda")
        else:
            device = torch.device(device_str)

        # ─── Multi-head 訓練設定 (N-head 可擴展) ────────────────────────────
        # recipe.model.multi_head → 自動：
        #   1. 注入 num_classes_aux 到 model_extras（保留現有 network constructor 簽名）
        #   2. 寫 multi_head.json 到 experiment_dir（trainer 讀此檔取得 loss weight 等）
        #   3. 設 MUTP_AUX_SEG_FOLDER 指向預處理好的 aux npz 資料夾
        #   4. 設 MUTP_MULTI_HEAD_CONFIG 指向 multi_head.json
        import pathlib as _pl
        import json as _json
        mh = getattr(m, "multi_head", None)
        if mh and getattr(mh, "enabled", False):
            if len(mh.heads) < 2:
                raise ValueError("multi_head.enabled=true 但 heads 少於 2 個")
            main_head = mh.heads[0]
            aux_heads = mh.heads[1:]
            # 1. 注入 num_classes_aux 給 network constructor
            #    (network 簽名: num_classes_aux=int; 只支援單 aux head 的場景)
            if len(aux_heads) > 1:
                print(f"[MUTP][WARN] {len(aux_heads)} aux heads 設定，但目前 network 只支援 1 aux head；用 heads[1]")
            r.model.extra = dict(r.model.extra or {})
            r.model.extra["num_classes_aux"] = aux_heads[0].num_classes
            print(f"[MUTP] multi-head: injected num_classes_aux={aux_heads[0].num_classes} (from heads[1].num_classes)")
            # 2. 寫 multi_head.json
            exp_path = _pl.Path(e.experiment_dir) if e.experiment_dir else None
            if exp_path:
                mh_dict = {
                    "enabled": True,
                    "heads": [
                        {
                            "name": h.name, "task": h.task,
                            "num_classes": h.num_classes,
                            "label_source": h.label_source,
                            "loss": {"type": h.loss.type, "weight": h.loss.weight, "params": dict(h.loss.params)},
                            "deep_supervision": h.deep_supervision,
                            "ignore_index": h.ignore_index,
                        }
                        for h in mh.heads
                    ],
                }
                mh_json = exp_path / "multi_head.json"
                mh_json.parent.mkdir(parents=True, exist_ok=True)
                mh_json.write_text(_json.dumps(mh_dict, indent=2, ensure_ascii=False))
                os.environ["MUTP_MULTI_HEAD_CONFIG"] = str(mh_json)
                print(f"[MUTP] multi-head: config → {mh_json}")
            # 3. aux seg location — 偵測佈局並設對應 env：
            #      inline → MUTP_AUX_SEG_INLINE=true   (新版扁平，{case}_aux_seg.npz 跟主檔同層)
            #      folder → MUTP_AUX_SEG_FOLDER=<dir>  (向後相容)
            _aux = _find_aux_seg_location(exp_path, e.source_dir, configuration)
            if _aux is None:
                print(f"[MUTP][WARN] multi-head 啟動，但 aux seg 找不到！"
                      f" 先跑：python3 scripts/generate_aux_seg.py --experiment-dir {exp_path}")
            elif _aux[0] == "inline":
                os.environ["MUTP_AUX_SEG_INLINE"] = "true"
                os.environ.pop("MUTP_AUX_SEG_FOLDER", None)
                print(f"[MUTP] multi-head: aux seg INLINE in {_aux[1]} ({{case}}_aux_seg.npz)")
            else:  # folder
                os.environ["MUTP_AUX_SEG_FOLDER"] = str(_aux[1])
                os.environ.pop("MUTP_AUX_SEG_INLINE", None)
                print(f"[MUTP] multi-head: aux seg FOLDER={_aux[1]} (legacy layout)")
        # 向後相容：舊 recipe 沒 multi_head:，但用 *_DualSegHead 架構 + extra.num_classes_aux
        elif m.architecture == "ResidualEncoderUNet_DualSegHead":
            exp_path = _pl.Path(e.experiment_dir) if e.experiment_dir else None
            _aux = _find_aux_seg_location(exp_path, e.source_dir, configuration)
            if _aux and _aux[0] == "inline":
                os.environ["MUTP_AUX_SEG_INLINE"] = "true"
                print(f"[MUTP] dual-head (legacy recipe): aux seg INLINE in {_aux[1]}")
            elif _aux:
                os.environ["MUTP_AUX_SEG_FOLDER"] = str(_aux[1])
                print(f"[MUTP] dual-head (legacy recipe): aux seg FOLDER={_aux[1]}")

        # Import nnU-Net run_training
        from nnunetv2.run.run_training import run_training

        print(f"[MUTP SegEngine] 啟動 nnU-Net 訓練")
        print(f"  Dataset: {dataset_id}, Config: {configuration}, Fold: {e.nnunet_fold}")
        print(f"  Trainer: {trainer_class}, Plans: {plans}")
        print(f"  Epochs: {t.num_epochs}, LR: {o.initial_lr}, Optimizer: {o.name}")
        if t.patch_size:
            print(f"  Patch size: {t.patch_size} (手動覆蓋)")
        if t.batch_size:
            print(f"  Batch size: {t.batch_size} (手動覆蓋)")

        # Phase 2.5: MLflow 背景同步 — 每個 epoch 完成即寫入
        mlflow_thread = None
        try:
            mlflow_thread = _start_mlflow_watcher(r, e.experiment_dir)
        except Exception as mlflow_err:
            print(f"[MUTP MLflow] 背景同步啟動失敗（非致命）：{mlflow_err}")

        run_training(
            dataset_name_or_id=str(dataset_id),
            configuration=configuration,
            fold=e.nnunet_fold,
            trainer_class_name=trainer_class,
            plans_identifier=plans,
            pretrained_weights=e.pretrained_weights,
            num_gpus=t.num_gpus,
            use_compressed_data=False,
            export_validation_probabilities=False,
            continue_training=t.continue_training,
            only_run_validation=t.validation_only,
            disable_checkpointing=t.disable_checkpointing,
            device=device,
            # 訓練參數
            initial_lr=o.initial_lr,
            oversample_foreground_percent=t.foreground_oversample_ratio,
            oversample_foreground_percent_val=t.val_foreground_oversample_ratio,
            num_iterations_per_epoch=t.iterations_per_epoch,
            num_epochs=t.num_epochs,
            optimizer_type=o.name,
            lr_scheduler_type=s.name,
            # 早停
            enable_early_stopping=es.enabled,
            early_stopping_patience=es.patience,
            early_stopping_min_delta=es.min_delta,
            # 取樣權重
            sampling_category_weights=sampling_weights,
            sampling_category_weight_mode=t.sampling_category_weight_mode,
            region_loss_weights=region_weights,
            normal_class_weights=normal_weights,
            enable_sampling_weights=t.enable_sampling_weights,
            enable_normal_upsample=t.enable_normal_upsample,
            # 進階
            enable_deep_supervision_logging=loss.deep_supervision_logging,
            enable_ema=t.enable_ema,
            ema_decay=t.ema_decay,
            cls_foreground_labels=cls_fg_labels,
            best_val_classes=best_val_cls,
            # 資料增強
            augmentation_config=r.augmentation.model_dump() if hasattr(r, 'augmentation') and r.augmentation else None,
            # Loss 設定（DC_and_CE_loss / Tversky_and_CE_loss 等）— 從 recipe.loss.components[0] 讀
            loss_config=_build_loss_config(r),
            # 模型額外建構參數（model.extra）— 例：S4/S5 的 spade_alpha_init
            model_extra_kwargs=dict(r.model.extra) if (hasattr(r.model, 'extra') and r.model.extra) else None,
            # 梯度累積
            enable_gradient_accumulation=t.enable_gradient_accumulation,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            # 由 MUTP watcher thread 追蹤 MLflow，關閉 Trainer 內建的 MLflow
            disable_builtin_mlflow=True,
            # 跳過 nnU-Net 內建的 perform_actual_validation()；MUTP 用 mutp inference + mutp eval 取代
            skip_final_validation=True,
        )

        # 停止 MLflow 背景同步，做最終 flush
        if mlflow_thread is not None:
            try:
                mlflow_thread.stop()
                mlflow_thread.join(timeout=30)
            except Exception as mlflow_err:
                print(f"[MUTP MLflow] 最終同步失敗（非致命）：{mlflow_err}")

        print(f"[MUTP SegEngine] 訓練完成")
        return {"status": "completed", "dataset_id": dataset_id}

    def run_plan_and_preprocess(self) -> None:
        """執行 nnUNetv2_plan_and_preprocess。"""
        r = self.recipe
        e = r.engine
        dataset_id = r.data.nnunet_dataset_id
        if dataset_id is None and e.experiment_dir:
            # 自動偵測：從 experiment_dir/nnUNet_raw/Dataset*_* 找 ID
            import pathlib
            raw_dir = pathlib.Path(e.experiment_dir) / "nnUNet_raw"
            if raw_dir.exists():
                ds_dirs = sorted(raw_dir.glob("Dataset*_*"))
                if ds_dirs:
                    # Dataset211_Aneurysm → 211
                    name = ds_dirs[0].name
                    try:
                        dataset_id = int(name.split("_")[0].replace("Dataset", ""))
                        print(f"[MUTP] 自動偵測 dataset_id={dataset_id}（from {name}）")
                    except ValueError:
                        pass
        if dataset_id is None:
            raise ValueError("需要 data.nnunet_dataset_id（recipe 未指定且無法從 experiment_dir 偵測）")

        configuration = r.data.nnunet_configuration or "3d_fullres"

        np_cores = getattr(r, '_np_cores', 8)  # 從 recipe 暫存的 np_cores 讀取

        # === 偵測 mask-fusion 架構 → 決定要不要 mask-aware preprocess ===
        # DeepConcat / SPADE 系列有 model.extra.mask_classes → 最後 1 個 channel 是 mask
        _model_arch = str(getattr(r.model, 'architecture', '') or '')
        _mask_fusion_archs = ('DeepConcat', 'SPADE', 'SPADEDecoder', 'SPADEDecoderAlpha', 'SPADEFull')
        _is_mask_fusion = any(a in _model_arch for a in _mask_fusion_archs)
        _mask_image_channels = None
        if _is_mask_fusion:
            # image_channels = num_input_channels - 1（最後 1 個是 mask）
            # 從 dataset.json 讀 channel_names 數量
            import json, pathlib
            raw_dir = pathlib.Path(e.experiment_dir) / "nnUNet_raw"
            ds_dirs = sorted(raw_dir.glob(f"Dataset{dataset_id:03d}_*"))
            if ds_dirs:
                dj_path = ds_dirs[0] / "dataset.json"
                if dj_path.exists():
                    dj = json.loads(dj_path.read_text())
                    _total_ch = len(dj.get('channel_names', {}))
                    if _total_ch >= 2:
                        _mask_image_channels = _total_ch - 1
                        print(f"[MUTP] mask-fusion arch detected ({_model_arch}); "
                              f"total_channels={_total_ch}, image_channels={_mask_image_channels}, "
                              f"mask ch={_mask_image_channels} 走 nearest-linear-onehot resample")

        env = self._get_subprocess_env()

        # === 分兩階段：plan (subprocess) → preprocess (in-process, mask-aware if needed) ===
        # Phase 1: plan only (--no_pp) via subprocess，確保 nnUNet CLI 相容性
        cmd_plan = [
            sys.executable, "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "-d", str(dataset_id),
            "-c", configuration,
            "-np", str(np_cores),
            "--no_pp",  # 只 plan，不 preprocess
        ]
        if e.gpu_memory_target != 8:
            cmd_plan.extend(["-gpu_memory_target", str(e.gpu_memory_target)])
        if e.overwrite_target_spacing:
            cmd_plan.extend(["-overwrite_target_spacing"] + [str(s) for s in e.overwrite_target_spacing])
        if e.overwrite_plans_name:
            cmd_plan.extend(["-overwrite_plans_name", e.overwrite_plans_name])
        if e.preprocessor_name:
            cmd_plan.extend(["-preprocessor_name", e.preprocessor_name])
        if e.verify_dataset_integrity:
            cmd_plan.append("--verify_dataset_integrity")

        print(f"[MUTP SegEngine] Phase 1 (plan only, subprocess): {' '.join(cmd_plan)}")
        result = subprocess.run(cmd_plan, timeout=3600, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"plan 失敗（exit code {result.returncode}）")

        # Phase 2: preprocess in-process，若 mask-fusion 就套 mask-aware wrapper
        # 環境變數要吃到 nnUNet_raw/preprocessed/results
        import os
        os.environ.update({k: v for k, v in env.items() if k.startswith('nnUNet_')})

        if _mask_image_channels is not None:
            from mutp.engines.backends.mask_preprocessor import (
                set_mask_image_channels, MaskAwareDefaultPreprocessor,
            )
            from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

            print(f"[MUTP SegEngine] Phase 2 (preprocess, in-process, mask-aware wrapper on)")
            # Monkey-patch ConfigurationManager.preprocessor_class → 我們的 MaskAware 版
            _orig_prop = ConfigurationManager.preprocessor_class
            ConfigurationManager.preprocessor_class = property(
                lambda _self: MaskAwareDefaultPreprocessor
            )
            set_mask_image_channels(_mask_image_channels)
            try:
                from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset
                # plans_identifier: 如果有 overwrite_plans_name 就用那個
                _plans_id = e.overwrite_plans_name or 'nnUNetPlans'
                preprocess_dataset(dataset_id, plans_identifier=_plans_id,
                                   configurations=(configuration,), num_processes=(np_cores,),
                                   verbose=False)
            finally:
                ConfigurationManager.preprocessor_class = _orig_prop
                set_mask_image_channels(None)
        else:
            print(f"[MUTP SegEngine] Phase 2 (preprocess, in-process, standard)")
            from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset
            _plans_id = e.overwrite_plans_name or 'nnUNetPlans'
            preprocess_dataset(dataset_id, plans_identifier=_plans_id,
                               configurations=(configuration,), num_processes=(np_cores,),
                               verbose=False)

        # 用 recipe 覆蓋 → 產出 plans variant
        self._apply_recipe_to_plans(dataset_id, configuration, recipe_path=getattr(r, '_recipe_path', None))

    def _apply_recipe_to_plans(self, dataset_id: int, configuration: str, recipe_path: str = None) -> None:
        """用 recipe 覆蓋 plans → 產出 variant 檔。"""
        import pathlib
        from mutp.engines.plans_utils import apply_recipe_to_plans, get_plans_variant_name

        r = self.recipe
        e = r.engine
        m = r.model
        t = r.training

        preprocessed_dir = pathlib.Path(e.experiment_dir) / "nnUNet_preprocessed"
        ds_dirs = list(preprocessed_dir.glob(f"Dataset{dataset_id:03d}_*"))
        if not ds_dirs:
            return
        plans_path = ds_dirs[0] / "nnUNetPlans.json"
        if not plans_path.exists():
            return

        # 梯度累積時，plans 的 batch_size = 每步實際 batch（recipe 寫總和，plans 取每步）
        # 不檢查 enable_gradient_accumulation flag — 只要 steps>1 一律分割，保持「recipe batch_size = 總和」語意
        plans_batch = t.batch_size
        if plans_batch and t.gradient_accumulation_steps and t.gradient_accumulation_steps > 1:
            plans_batch = (plans_batch + t.gradient_accumulation_steps - 1) // t.gradient_accumulation_steps
            print(f"[MUTP] batch_size {t.batch_size} ÷ {t.gradient_accumulation_steps} = {plans_batch}（plans 寫入每步實際 batch；recipe batch_size 表總和）")

        # 直接覆蓋 nnUNetPlans.json（每個 experiment 有自己的實體檔）
        modified = apply_recipe_to_plans(
            plans_path=plans_path,
            output_path=plans_path,  # 原地覆蓋
            configuration=configuration,
            architecture=m.architecture,
            patch_size=t.patch_size,
            batch_size=plans_batch,
            depth=getattr(m, 'depth', None) or 5,
        )

        print(f"[MUTP] {plans_path.name} 已更新：")
        for m_item in modified:
            print(f"  {m_item}")

        # 自動畫模型架構圖 → 給使用者驗證 plans.json 真的構出對的網路
        try:
            from mutp.engines.model_viz import visualize_model_architecture
            dataset_json = ds_dirs[0] / "dataset.json"
            if dataset_json.exists():
                visualize_model_architecture(
                    plans_path=str(plans_path),
                    dataset_json_path=str(dataset_json),
                    configuration=configuration,
                    output_dir=str(pathlib.Path(e.experiment_dir)),
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[MUTP] model_viz 失敗（跳過）：{type(exc).__name__}: {exc}")

    def run_inference(self, input_dir: str, output_dir: str, mask_dir: str | None = None) -> None:
        """執行 nnU-Net 推論 — 直接呼叫 predict_from_raw_data（不用 subprocess）。

        mask_dir：若提供，直接用此目錄當 sliding window mask，覆寫 recipe.inference.inference_mask
        的 {name}Ts 自動解析（HNM 訓練集推論用）。
        """
        # 過濾 batchgenerators MultiThreadedAugmenter daemon thread 在 process 結束時的 cleanup 錯誤
        import threading as _threading
        import traceback as _traceback
        _orig_excepthook = _threading.excepthook
        def _filter_hook(args):
            if args.exc_type is FileNotFoundError and args.exc_traceback is not None:
                tb_text = "".join(_traceback.format_tb(args.exc_traceback))
                if "results_loop" in tb_text or "multi_threaded_augmenter" in tb_text:
                    return
            _orig_excepthook(args)
        _threading.excepthook = _filter_hook

        r = self.recipe
        e = r.engine
        t = r.training
        inf = r.inference

        dataset_id = r.data.nnunet_dataset_id
        if dataset_id is None and e.experiment_dir:
            raw_dir = pathlib.Path(e.experiment_dir) / "nnUNet_raw"
            if raw_dir.exists():
                ds_dirs = sorted(raw_dir.glob("Dataset*_*"))
                if ds_dirs:
                    try:
                        dataset_id = int(ds_dirs[0].name.split("_")[0].replace("Dataset", ""))
                    except ValueError:
                        pass
        if dataset_id is None:
            raise ValueError("需要 data.nnunet_dataset_id（無法從 experiment_dir 偵測）")
        configuration = r.data.nnunet_configuration or "3d_fullres"

        # 設定 nnU-Net 環境變數
        self._setup_env_vars()

        # model_folder: nnUNet_results/DatasetXXX_Name/TrainerClass__Plans__Config
        results_dir = pathlib.Path(e.experiment_dir) / "nnUNet_results"
        ds_dirs = sorted(results_dir.glob(f"Dataset{dataset_id:03d}_*")) if results_dir.exists() else []
        if not ds_dirs:
            raise FileNotFoundError(f"找不到 Dataset{dataset_id:03d}_* in {results_dir}")

        trainer_class = e.nnunet_trainer_class or "nnUNetTrainer"
        plans = e.nnunet_plans or "nnUNetPlans"
        model_folder = ds_dirs[0] / f"{trainer_class}__{plans}__{configuration}"
        if not model_folder.exists():
            raise FileNotFoundError(f"model_folder 不存在：{model_folder}")

        # 把 recipe 的 best/latest/final shortcut 轉成 nnU-Net 真實檔名（checkpoint_X.pth）
        # 如果使用者填完整檔名（含 .pth），直接用
        _chk = inf.checkpoint or "best"
        if _chk in ("best", "latest", "final"):
            chk_name = f"checkpoint_{_chk}.pth"
        elif not _chk.endswith(".pth"):
            chk_name = f"checkpoint_{_chk}.pth"
        else:
            chk_name = _chk

        # GPU — 優先 inference.desired_gpu_index → training.device
        gpu_index = inf.desired_gpu_index
        if gpu_index is None:
            device_str = t.device if t.device else "cuda:0"
            if ":" in device_str:
                gpu_index = int(device_str.split(":")[1])
            else:
                gpu_index = 0

        # fold
        folds = inf.ensemble_folds if inf.ensemble_folds else (e.nnunet_fold,)

        # device — CUDA_VISIBLE_DEVICES 選好 GPU 後，device index 固定為 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        device = torch.device("cuda")
        _desired_gpu_for_predict = 0  # CUDA_VISIBLE_DEVICES 已選 GPU，predict 內部用 index=0

        print(f"[MUTP SegEngine] 執行推論")
        print(f"  model:      {model_folder.name}")
        print(f"  checkpoint: {chk_name}")
        print(f"  GPU:        cuda:{gpu_index}")
        print(f"  input:      {input_dir}")
        print(f"  output:     {output_dir}")
        print(f"  step_size:  {inf.step_size}")
        print(f"  batch_size: {inf.batch_size}")
        print(f"  TTA:        {'OFF' if inf.disable_tta else 'ON'}")
        print(f"  gaussian:   {'ON' if inf.gaussian_weighting else 'OFF'}")
        print(f"  save_prob:  {'ON' if inf.output_probabilities else 'OFF'}")

        # PyTorch 2.6+ 預設 weights_only=True，nnU-Net checkpoint 含 numpy 物件會報錯
        import numpy as np
        _np_safe = [np._core.multiarray.scalar, np.dtype]
        for _name in ["Float16DType", "Float32DType", "Float64DType",
                      "Int8DType", "Int16DType", "Int32DType", "Int64DType",
                      "UInt8DType"]:
            if hasattr(np.dtypes, _name):
                _np_safe.append(getattr(np.dtypes, _name))
        torch.serialization.add_safe_globals(_np_safe)

        # 使用自訂版 predict_from_raw_data（含 Mask, batch_size, desired_gpu_index）
        from mutp.engines.backends.custom_predict import predict_from_raw_data

        # inference_mask → nnUNet_raw/.../normalsTs 目錄
        # 若 caller 已透過 mask_dir 顯式傳入路徑，優先用，跳過 recipe 自動解析
        if mask_dir:
            print(f"  mask (override): {mask_dir}")
        elif inf.inference_mask:
            raw_dir = pathlib.Path(e.experiment_dir) / "nnUNet_raw"
            for raw_ds in sorted(raw_dir.glob(f"Dataset{dataset_id:03d}_*")) if raw_dir.exists() else []:
                mask_candidate = raw_ds / f"{inf.inference_mask}Ts"
                # 目錄空的（例如遠端 tarball 沒把 test mask 帶進來）也視為 not found → 走 fallback
                if mask_candidate.is_dir() and any(mask_candidate.iterdir()):
                    mask_dir = str(mask_candidate)
                    print(f"  mask:       {mask_dir}")
                    break
            # Fallback: external test dataset — imagesTs 旁邊常有 masks/{vessel,brain}_mask/
            # 決定 sub-key：優先 recipe.evaluation.iso_fp_by_region.mask_key，否則 vessel_mask
            if not mask_dir and input_dir:
                _in_p = pathlib.Path(input_dir)
                _ds_root = _in_p.parent if _in_p.name.startswith("imagesT") else _in_p
                _mk = "vessel_mask"
                try:
                    _iso = r.evaluation.region_segmentation.iso_fp_by_region
                    if _iso and getattr(_iso, 'mask_key', None):
                        _mk = _iso.mask_key
                except AttributeError:
                    pass
                _cand = _ds_root / "masks" / _mk
                if _cand.exists() and any(_cand.iterdir()):
                    mask_dir = str(_cand)
                    print(f"  mask (external test fallback): {mask_dir}")

        predict_from_raw_data(
            input_dir,
            mask_dir,
            output_dir,
            str(model_folder),
            use_folds=folds,
            tile_step_size=inf.step_size,
            use_gaussian=inf.gaussian_weighting,
            use_mirroring=not inf.disable_tta,
            perform_everything_on_gpu=True,
            verbose=True,
            save_probabilities=inf.output_probabilities,
            overwrite=not inf.continue_prediction,
            checkpoint_name=chk_name,
            num_processes_preprocessing=inf.num_preprocessing_processes,
            num_processes_segmentation_export=inf.num_segmentation_processes,
            folder_with_segs_from_prev_stage=inf.prev_stage_predictions,
            num_parts=inf.num_parts if inf.num_parts > 1 else 1,
            part_id=inf.part_id,
            desired_gpu_index=_desired_gpu_for_predict,
            device=device,
            batch_size=inf.batch_size,
            apply_mask_to_prediction=inf.apply_mask_to_prediction,
        )

        # 強制清理 MultiThreadedAugmenter 的 background threads/processes
        # 避免 "FileNotFoundError: No such file or directory" cleanup warning
        import gc
        import time as _gc_time
        gc.collect()
        _gc_time.sleep(0.3)  # 給 thread 時間乾淨結束

        print(f"[MUTP SegEngine] 推論完成")

    def _get_subprocess_env(self) -> dict:
        """取得 subprocess 用的環境變數（含 nnU-Net 路徑）。"""
        env = os.environ.copy()
        if self._nnunet_env:
            env.update(self._nnunet_env)
        return env

    def get_nnunet_raw_dir(self) -> str:
        """取得 nnUNet_raw 目錄路徑。"""
        if self._nnunet_env:
            return self._nnunet_env["nnUNet_raw"]
        return os.environ.get("nnUNet_raw", "")