#!/usr/bin/env python
# coding: utf-8

# ## 測試nnUNet Inference code  
# 因為sliding_window用林君彥的pipeline要1個case inference 9000多pitch 要30分鐘，所以要改

# In[1]:


import inspect
import multiprocessing
import os
import shutil
import traceback
from asyncio import sleep
from copy import deepcopy
from typing import Tuple, Union, List

import nnunetv2
import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs,     save_json
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_softmax
#from nnunetv2.inference.sliding_window_prediction import predict_sliding_window_return_logits, compute_gaussian
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.file_path_utilities import get_output_folder, should_i_save_to_file, check_workers_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels, convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu, gaussian, threshold_otsu, frangi
from skimage.measure import label, regionprops, regionprops_table
import time


# In[2]:


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]], list_of_segs_from_prev_stage_files: Union[List[None], List[str]],
                 preprocessor: DefaultPreprocessor, output_filenames_truncated: List[str],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1):
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json =             preprocessor, plans_manager, configuration_manager, dataset_json

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        files = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        ofile = self._data[idx][2]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, data_properites = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)
        #if seg_prev_stage is not None:
        #    seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
        #    data = np.vstack((data, seg_onehot))

        if np.prod(data.shape) > (2e9 / 4 * 0.85):
            # we need to temporarily save the preprocessed image due to process-process communication restrictions
            np.save(ofile + '.npy', data)
            data = ofile + '.npy'

        return {'data': data, 'seg': seg, 'data_properites': data_properites, 'ofile': ofile}


# In[3]:


def load_what_we_need(model_training_output_dir, use_folds, checkpoint_name):
    # we could also load plans and dataset_json from the init arguments in the checkpoint. Not quite sure what is the
    # best method so we leave things as they are for the moment.
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    
    plans_manager = PlansManager(plans)

    if isinstance(use_folds, str):
        use_folds = [use_folds]

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'), weights_only=False)
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if                 'inference_allowed_mirroring_axes' in checkpoint.keys() else None
            # MUTP: env var override for TTA axis subset (for benchmarking TTA-4 vs TTA-8 etc.)
            _env_axes = os.environ.get('MUTP_MIRROR_AXES_SUBSET', None)
            if _env_axes:
                try:
                    _override = tuple(int(x) for x in _env_axes.split(',') if x.strip() != '')
                    print(f"[MUTP] mirror axes overridden by MUTP_MIRROR_AXES_SUBSET: "
                          f"{inference_allowed_mirroring_axes} → {_override}")
                    inference_allowed_mirroring_axes = _override
                except Exception as _e:
                    print(f"[MUTP] MUTP_MIRROR_AXES_SUBSET parse failed ({_e}), using checkpoint default.")

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')

    # MUTP: infer model.extra kwargs from checkpoint tensor shapes (checkpoint is the ground truth,
    # recipe.model.extra may have drifted since training or default class attrs may leak).
    _first_state = parameters[0] if parameters else {}
    _existing = getattr(trainer_class, 'MODEL_EXTRA_KWARGS', None) or {}

    # (a) num_classes_aux from aux_seg_layers (DualSegHead only)
    _aux_key = 'aux_seg_layers.0.weight'
    if _aux_key in _first_state:
        _num_aux = int(_first_state[_aux_key].shape[0])
        _existing['num_classes_aux'] = _num_aux
        print(f'[MUTP inference] auto-detected num_classes_aux={_num_aux} from checkpoint aux_seg_layers')

    # (b) mask_classes from enc_fuse (DeepConcat) or dec_spade.shared (SPADE variants)
    #   DeepConcat: enc_fuse.0.weight shape = [features, features + mask_classes, 1, 1, 1]
    #   SPADE:      dec_spade.0.shared.weight shape = [hidden, mask_classes, 3, 3, 3]
    _num_mc = None
    if 'enc_fuse.0.weight' in _first_state:
        _w = _first_state['enc_fuse.0.weight']
        _num_mc = int(_w.shape[1] - _w.shape[0])
    elif 'dec_spade.0.shared.weight' in _first_state:
        _num_mc = int(_first_state['dec_spade.0.shared.weight'].shape[1])
    if _num_mc is not None:
        _existing['mask_classes'] = _num_mc
        print(f'[MUTP inference] auto-detected mask_classes={_num_mc} from checkpoint')

    # (c) classifier_num_classes from classifier head weight
    #   AttentionClassifier: classifier_head.pooling.classifier.weight
    #   GuidedClassifier:    classifier_head.classifier.weight
    #   Both shape = [num_classes, feature_dim]
    for _cls_key in ('classifier_head.pooling.classifier.weight',
                     'classifier_head.classifier.weight'):
        if _cls_key in _first_state:
            _n_cls = int(_first_state[_cls_key].shape[0])
            _existing['classifier_num_classes'] = _n_cls
            print(f'[MUTP inference] auto-detected classifier_num_classes={_n_cls} from {_cls_key}')
            break

    trainer_class.MODEL_EXTRA_KWARGS = _existing

    network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision=False)
    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network, trainer_name


# In[4]:


def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
    print('use_folds is None, attempting to auto detect available folds')
    fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
    fold_folders = [i for i in fold_folders if i != 'fold_all']
    fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
    use_folds = [int(i.split('_')[-1]) for i in fold_folders]
    print(f'found the following folds: {use_folds}')
    return use_folds


# In[5]:


import warnings

import numpy as np
import torch
from typing import Union, Tuple, List
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter
from torch import nn

from nnunetv2.utilities.helpers import empty_cache, dummy_context


# In[6]:


def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8, dtype=np.float16)         -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


# In[7]:


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) ->         List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


# In[8]:


def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float,
                                 verbose: bool = False):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) '                                                       'must be one shorter than len(image_size) (only dimension '                                                       'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
        if verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
        if verbose: print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    yield slicer


# In[ ]:


def _forward_network(network: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """呼叫 network，兼容三種輸出：

    - 單輸出 seg：vanilla / SPADE / DeepConcat → 回 (seg, zeros)
    - (seg, cls)：classifier head → cls 是 [B, num_cls] 2-D → 回 (seg, softmax(cls)[:,1])
    - (seg_main, seg_aux)：dual seg head (multi-task) → aux 是 [B, C, ...] N-D，
      推論只看 main，aux 忽略 → 回 (seg_main, zeros)
    """
    output = network(x)
    if isinstance(output, (tuple, list)):
        prediction, second = output
        # 用 dim 區分 cls (2-D) vs aux seg (>=4-D)
        if second.dim() <= 2:
            cls_prob = torch.softmax(second, dim=1)[:, 1]  # shape: [batch_size]
        else:
            # aux seg head — inference 不用
            cls_prob = torch.zeros(x.shape[0], device=x.device)
    else:
        prediction = output
        cls_prob = torch.zeros(x.shape[0], device=x.device)
    return prediction, cls_prob


def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None)         -> Tuple[torch.Tensor, torch.Tensor]:
    prediction, cls_prob = _forward_network(network, x)

    if mirror_axes is not None:
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        cls_prob_sum = cls_prob.clone()

        for axes in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]:
            if all(a - 2 in mirror_axes for a in axes):
                pred_tmp, cls_tmp = _forward_network(network, torch.flip(x, axes))
                prediction += torch.flip(pred_tmp, axes)
                cls_prob_sum += cls_tmp

        prediction /= num_predictons
        cls_prob = cls_prob_sum / num_predictons

    return prediction, cls_prob


# ## 最需要改的地方

# In[ ]:


import torch.nn.functional as F

def predict_sliding_window_return_logits(network: nn.Module,
                                         input_image: Union[np.ndarray, torch.Tensor],
                                         vessel_image: Union[np.ndarray, torch.Tensor],
                                         num_segmentation_heads: int,
                                         tile_size: Tuple[int, ...],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         precomputed_gaussian: torch.Tensor = None,
                                         perform_everything_on_gpu: bool = True,
                                         verbose: bool = True,
                                         device: torch.device = torch.device('cuda'),
                                         batch_size: int = 1,
                                         use_sigmoid: bool = False,
                                         apply_mask_to_prediction: bool = False) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    if perform_everything_on_gpu:
        assert device.type == 'cuda', 'Can use perform_everything_on_gpu=True only when device="cuda"'

    network = network.to(device)
    network.eval()

    empty_cache(device)
    
    with torch.no_grad():
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if not torch.cuda.is_available():
                if perform_everything_on_gpu:
                    print('WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...')
                perform_everything_on_gpu = False

            results_device = device if perform_everything_on_gpu else torch.device('cpu')

            if verbose: print("step_size:", tile_step_size)
            if verbose: print("mirror_axes:", mirror_axes)

            if not isinstance(input_image, torch.Tensor):
                # pytorch will warn about the numpy array not being writable. This doesnt matter though because we
                # just want to read it. Suppress the warning in order to not confuse users...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_image = torch.from_numpy(input_image)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True, None)
            #data_vessel, slicer_revert_padding = pad_nd_image(vessel_image, data.shape, 'constant', {'value': 0}, True, None)
            
            # 計算需要補齊的大小:tensorA = torch.randn(1, 127, 512, 512)  # 假設是 tensorA
            pad_height = (data.shape[2] - vessel_image.shape[2])  # 需要補齊的高度 (上和下)
            pad_width = (data.shape[3] - vessel_image.shape[3])  # 需要補齊的寬度 (左和右)
            pad_depth = (data.shape[1] - vessel_image.shape[1])  # 需要補齊的深度 (前和後)

            # 計算每一維的補齊值
            # pad順序為 (左, 右, 上, 下, 前, 後)
            padding = (pad_width // 2, pad_width - pad_width // 2,  # 深度（前後）
                       pad_height // 2, pad_height - pad_height // 2,  # 高度（上下）
                       pad_depth, 0)  # 寬度（左右）
            
            # 使用 F.pad 進行補齊
            data_vessel = F.pad(vessel_image, padding, mode='constant', value=0)
                        
            print("step_size:", tile_step_size) #0.5 => 步距重疊率
            print("mirror_axes:", mirror_axes) #還是0
            print('data pad後的大小:', data.shape) #這邊還是512
            print('data_vessel pad後的大小:', data_vessel.shape) #這邊還是512

            if use_gaussian:
                gaussian = torch.from_numpy(
                    compute_gaussian(tile_size, sigma_scale=1. / 8)) if precomputed_gaussian is None else precomputed_gaussian
                gaussian = gaussian.half()
                # make sure nothing is rounded to zero or we get division by zero :-(
                mn = gaussian.min()
                if mn == 0:
                    gaussian.clip_(min=mn)
            else:
                # 不使用 gaussian 時，設置為 None 以節省記憶體
                gaussian = None
                    
            slicers = get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size, verbose=verbose)

            # preallocate results and num_predictions. Move everything to the correct device
            try:
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                # 新增：分類機率圖（不需要多個 heads，只需要單一維度）
                classification_prob_map = torch.zeros(data.shape[1:], dtype=torch.half,
                                                     device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                # 分類機率圖的計數器（不使用高斯加權）
                n_predictions_cls = torch.zeros(data.shape[1:], dtype=torch.half,
                                               device=results_device)
                if use_gaussian and gaussian is not None:
                    gaussian = gaussian.to(results_device)
            except RuntimeError:
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                results_device = torch.device('cpu')
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                classification_prob_map = torch.zeros(data.shape[1:], dtype=torch.half,
                                                     device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                n_predictions_cls = torch.zeros(data.shape[1:], dtype=torch.half,
                                               device=results_device)
                if use_gaussian and gaussian is not None:
                    gaussian = gaussian.to(results_device)
            finally:
                empty_cache(device)

            if use_gaussian:
                # 優化：先把 vessel mask 搬到 CPU，用 numpy 做 slicer filter（避免 GPU kernel launch overhead）
                slicers_list = list(slicers)
                vessel_cpu = (data_vessel[0] > 0).cpu().numpy()

                patches_to_process = []
                slicers_to_process = []
                for sl in slicers_list:
                    if vessel_cpu[sl[1:]].any():
                        patches_to_process.append(data[sl][None])
                        slicers_to_process.append(sl)

                if len(patches_to_process) > 0:
                    if verbose:
                        print(f"[Gaussian模式] 處理 {len(patches_to_process)} 個有血管的 patches，使用 batch_size={batch_size}")

                    for i in range(0, len(patches_to_process), batch_size):
                        batch_end = min(i + batch_size, len(patches_to_process))
                        batch_patches = patches_to_process[i:batch_end]
                        batch_slicers = slicers_to_process[i:batch_end]

                        batch_tensor = torch.cat(batch_patches, dim=0).to(device, non_blocking=False)
                        batch_predictions, batch_cls_probs = maybe_mirror_and_predict(network, batch_tensor, mirror_axes)
                        # 根據 use_sigmoid 決定 activation
                        # False (預設): softmax → 互斥 class（單一像素只屬於一類）
                        # True: sigmoid → 重疊 class（region-based，一個像素可屬多類）
                        if use_sigmoid:
                            batch_predictions = torch.sigmoid(batch_predictions).to(results_device)
                        else:
                            batch_predictions = torch.softmax(batch_predictions, dim=1).to(results_device)
                        batch_cls_probs = batch_cls_probs.to(results_device)

                        for j, sl in enumerate(batch_slicers):
                            predicted_logits[sl] += batch_predictions[j] * gaussian
                            n_predictions[sl[1:]] += gaussian
                            classification_prob_map[sl[1:]] += batch_cls_probs[j]
                            n_predictions_cls[sl[1:]] += 1
            else:
                # 不使用 gaussian 權重的情況：可以完全跳過沒有血管的區域，大幅加速
                patches_to_process = []
                slicers_to_process = []
                all_slicers = list(slicers)  # 先轉換成 list 以便重複使用
                
                # 只收集有血管的 patches，完全跳過空白區域
                for sl in all_slicers:
                    if torch.sum(data_vessel[sl]) > 0:
                        workon = data[sl][None]
                        patches_to_process.append(workon)
                        slicers_to_process.append(sl)
                
                if len(patches_to_process) > 0:
                    if verbose:
                        total_patches = len(all_slicers)
                        print(f"[非Gaussian模式] 跳過 {total_patches - len(patches_to_process)} 個空白 patches，只處理 {len(patches_to_process)} 個有血管的 patches，使用 batch_size={batch_size}")
                    
                    for i in range(0, len(patches_to_process), batch_size):
                        batch_end = min(i + batch_size, len(patches_to_process))
                        batch_patches = patches_to_process[i:batch_end]
                        batch_slicers = slicers_to_process[i:batch_end]
                        
                        # 將 batch 中的 patches 組合成一個 tensor
                        batch_tensor = torch.cat(batch_patches, dim=0).to(device, non_blocking=False)
                        
                        # 批次預測
                        start_time_batch = time.time()
                        batch_predictions, batch_cls_probs = maybe_mirror_and_predict(network, batch_tensor, mirror_axes)
                        batch_predictions = batch_predictions.to(results_device)
                        batch_cls_probs = batch_cls_probs.to(results_device)
                        print(f"[Done] maybe_mirror_and_predict no. {i} spend {time.time() - start_time_batch:.3f} sec")
                        
                        # 處理每個預測結果
                        for j, (prediction, cls_prob, sl) in enumerate(zip(batch_predictions, batch_cls_probs, batch_slicers)):
                            # softmax (互斥) 或 sigmoid (重疊 region-based)
                            if use_sigmoid:
                                prediction = torch.sigmoid(prediction)
                            else:
                                prediction = torch.softmax(prediction, 0)

                            # 不使用高斯權重，直接累加
                            predicted_logits[sl] += prediction
                            n_predictions[sl[1:]] += 1
                            
                            # 處理分類機率圖：整個 patch 填入該機率值，不使用高斯加權
                            classification_prob_map[sl[1:]] += cls_prob.item()
                            n_predictions_cls[sl[1:]] += 1
                
                # 對於完全沒有血管的區域，設置一個預設的背景預測
                # 這樣可以避免這些區域保持未初始化狀態
                if len(patches_to_process) < len(all_slicers):
                    # 創建背景預測：[1.0, 0.0] 表示背景類別的機率為 1
                    background_prediction = torch.zeros((num_segmentation_heads, *tile_size), device=results_device)
                    background_prediction[0] = 1.0  # 背景類別設為 1
                    
                    for sl in all_slicers:
                        if torch.sum(data_vessel[sl]) == 0:  # 沒有血管的區域
                            predicted_logits[sl] += background_prediction
                            n_predictions[sl[1:]] += 1

            # 安全除法，避免除以零產生 NaN
            # 對於 n_predictions 為 0 的位置，保持 predicted_logits 為 0
            mask = n_predictions > 0
            predicted_logits = torch.where(mask.unsqueeze(0), 
                                         predicted_logits / n_predictions.unsqueeze(0), 
                                         predicted_logits)
            
            # 處理分類機率圖的平均化
            mask_cls = n_predictions_cls > 0
            classification_prob_map = torch.where(mask_cls,
                                                 classification_prob_map / n_predictions_cls,
                                                 classification_prob_map)
            
            if verbose:
                zero_predictions = torch.sum(n_predictions == 0).item()
                total_voxels = torch.numel(n_predictions)
                if zero_predictions > 0:
                    print(f"警告：有 {zero_predictions}/{total_voxels} 個體素沒有被任何 patch 覆蓋到")
                    print(f"這些位置將保持為零值（通常是影像邊緣或完全沒有血管的區域）")
                
                print(f"分類機率圖統計 - 最小值: {classification_prob_map.min().item():.4f}, "
                      f"最大值: {classification_prob_map.max().item():.4f}, "
                      f"平均值: {classification_prob_map.mean().item():.4f}")
            #print('predicted_logits.shape:', predicted_logits.shape, ' data_vessel.shape:', data_vessel.shape)
            #predicted_logits.shape: torch.Size([2, 127, 512, 512])  data_vessel.shape: torch.Size([1, 127, 512, 512])
            
            #只與vessel相乘
            #print('predicted_logits.shape:', predicted_logits.shape, ' data_vessel.shape:', data_vessel.shape)
            #predicted_logits = predicted_logits[1, :, :, :] * data_vessel.to(results_device)
            #print('predicted_logits.shape:', predicted_logits.shape)
            # 使用 unsqueeze 增加一個維度，放在最前面
            #predicted_logits = predicted_logits[0, :, :, :].unsqueeze(0)
            #predicted_logits = data_vessel.to(results_device)
                        
            # 與 vessel/mask 相乘（可設定；aneurysm 預設 True 用 vessel gating 去 FP；infarct 預設 False）
            # 內部強制二值化 data_vessel，避免多類值 scale-up prob；原多類 mask 仍保留在 mask 檔供 iso_fp 分析
            if apply_mask_to_prediction:
                _mask_bin = (data_vessel > 0).to(data_vessel.dtype)
                repeat_vessel = _mask_bin.repeat(num_segmentation_heads, 1, 1, 1)
                predicted_logits = predicted_logits * repeat_vessel.to(results_device)
                if verbose:
                    print('[apply_mask] multiplied prediction by binarized data_vessel')
            else:
                if verbose:
                    print('[apply_mask] skipped (apply_mask_to_prediction=False)')
            #predicted_logits = repeat_vessel.to(results_device)            

    empty_cache(device)
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])], classification_prob_map[tuple(slicer_revert_padding[1:])]


# In[11]:


from typing import Tuple, Union, List
import numpy as np
#from nibabel import io_orientation

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel as nib

def write_probabilities(seg, output_fname, img_nii):
    # revert transpose — 支援 3D (single-label) 和 4D (multi-label)
    if seg.ndim == 3:
        seg = seg.transpose((2, 1, 0)).astype(np.float32)
    elif seg.ndim == 4:
        # (n_channels, D, H, W) → (W, H, D, n_channels) for NIfTI 4D
        seg = seg.transpose((3, 2, 1, 0)).astype(np.float32)

    affine = img_nii.affine
    header = img_nii.header.copy()
    new_nii = nib.nifti1.Nifti1Image(seg, affine, header=header)

    nib.save(new_nii, output_fname)


# ## 最需要改的地方2

# In[ ]:


import os
from copy import deepcopy
from typing import Union, List

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

def export_prediction_probabilities(predicted_array_or_file: Union[np.ndarray, str], 
                                    classification_prob_array_or_file: Union[np.ndarray, str],
                                    properties_dict: dict,
                                    vessel_image, img_nii,
                                    configuration_manager: ConfigurationManager,
                                    plans_manager: PlansManager,
                                    dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                    save_probabilities: bool = False):
    
    if isinstance(predicted_array_or_file, str):
        tmp = deepcopy(predicted_array_or_file)
        if predicted_array_or_file.endswith('.npy'):
            predicted_array_or_file = np.load(predicted_array_or_file)
        elif predicted_array_or_file.endswith('.npz'):
            predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
        os.remove(tmp)
    
    # 處理分類機率圖
    if isinstance(classification_prob_array_or_file, str):
        tmp_cls = deepcopy(classification_prob_array_or_file)
        if classification_prob_array_or_file.endswith('.npy'):
            classification_prob_array_or_file = np.load(classification_prob_array_or_file)
        os.remove(tmp_cls)

    predicted_array_or_file = predicted_array_or_file.astype(np.float32)
    print('before')
    print('predicted_array_or_file.shape:', predicted_array_or_file.shape)
    print('np.max(predicted_array_or_file):', np.max(predicted_array_or_file))
    print('np.median(predicted_array_or_file):', np.median(predicted_array_or_file))

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if         len(configuration_manager.spacing) ==         len(properties_dict['shape_after_cropping_and_before_resampling']) else         [properties_dict['spacing'][0], *configuration_manager.spacing]
    
    print('properties_dict[shape_after_cropping_and_before_resampling]:', properties_dict['shape_after_cropping_and_before_resampling'])
    print('current_spacing:', current_spacing)
    print('properties_dict[spacing]:', properties_dict['spacing'])
    
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted_array_or_file,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    
    print('after')
    print('predicted_array_or_file.shape:', predicted_array_or_file.shape)
    print('np.max(predicted_array_or_file):', np.max(predicted_array_or_file))
    print('np.median(predicted_array_or_file):', np.median(predicted_array_or_file))    
    
    
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    """
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)

    # put result in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation
    print('segmentation_reverted_cropping.shape:', segmentation_reverted_cropping.shape)

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    """
    
    # save
    # probabilities are already resampled

    # apply nonlinearity
#     predicted_array_or_file = label_manager.apply_inference_nonlin(predicted_array_or_file)

#     print('apply nonlinearity')
#     print('predicted_array_or_file.shape:', predicted_array_or_file.shape)
#     print('np.max(predicted_array_or_file):', np.max(predicted_array_or_file))
#     print('np.median(predicted_array_or_file):', np.median(predicted_array_or_file))
    
    # revert cropping
    probs_reverted_cropping = label_manager.revert_cropping(predicted_array_or_file,
                                                            properties_dict['bbox_used_for_cropping'],
                                                            properties_dict['shape_before_cropping'])
    
    print('revert cropping')
    print('probs_reverted_cropping.shape:', probs_reverted_cropping.shape)
    print('np.max(probs_reverted_cropping):', np.max(probs_reverted_cropping))
    print('np.median(probs_reverted_cropping):', np.median(probs_reverted_cropping))
    
    # 去掉背景 channel（channel 0），保留所有前景 channel
    # single-label: (2, D, H, W) → (1, D, H, W)
    # multi-label:  (3, D, H, W) → (2, D, H, W)  e.g. CMB + Uncertain
    probs_reverted_cropping = probs_reverted_cropping[1:]  # shape: (n_fg, D, H, W)
    print(f'foreground channels: {probs_reverted_cropping.shape[0]}')

    if probs_reverted_cropping is None:
        raise ValueError("Reverting cropping failed, 'probs_reverted_cropping' is None.")

    # revert transpose
    probs_reverted_cropping = probs_reverted_cropping.transpose([0] + [i + 1 for i in
                                                                plans_manager.transpose_backward])

    # 寫入 NIfTI
    output_file = output_file_truncated + dataset_json_dict_or_file['file_ending']
    if probs_reverted_cropping.shape[0] == 1:
        # single-label: 寫 3D（與原始行為一致）
        write_probabilities(probs_reverted_cropping[0], output_file, img_nii)
    else:
        # multi-label: 寫 4D（每個前景 class 一個 channel）
        write_probabilities(probs_reverted_cropping, output_file, img_nii)
    
    # ========== 處理分類機率圖（僅有 classifier head 的模型才有非零值）==========
    if np.max(classification_prob_array_or_file) > 0:
        print('\n處理分類機率圖...')
        classification_prob_array_or_file = classification_prob_array_or_file.astype(np.float32)
        classification_prob_with_channel = np.expand_dims(classification_prob_array_or_file, axis=0)
        classification_prob_resampled = configuration_manager.resampling_fn_probabilities(
            classification_prob_with_channel,
            properties_dict['shape_after_cropping_and_before_resampling'],
            current_spacing,
            properties_dict['spacing'])
        cls_prob_reverted_cropping = label_manager.revert_cropping(
            classification_prob_resampled,
            properties_dict['bbox_used_for_cropping'],
            properties_dict['shape_before_cropping'])
        cls_prob_reverted_cropping = cls_prob_reverted_cropping.transpose(
            [0] + [i + 1 for i in plans_manager.transpose_backward])
        output_cls_file = output_file_truncated + '_classification_prob' + dataset_json_dict_or_file['file_ending']
        write_probabilities(cls_prob_reverted_cropping[0,:,:,:], output_cls_file, img_nii)
        print(f'分類機率圖已儲存至: {output_cls_file}')
    else:
        print('\n無 classifier head，跳過分類機率圖')


# In[ ]:


def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          Mask_list_of_lists_or_Mask_folder: Union[str, List[List[str]]],
                          output_folder: str,
                          model_training_output_dir: str,
                          use_folds: Union[Tuple[int, ...], str] = None,
                          tile_step_size: float = 0.5,
                          use_gaussian: bool = True,
                          use_mirroring: bool = True,
                          perform_everything_on_gpu: bool = True,
                          verbose: bool = True,
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          checkpoint_name: str = 'checkpoint_final.pth',
                          num_processes_preprocessing: int = default_num_processes,
                          num_processes_segmentation_export: int = default_num_processes,
                          folder_with_segs_from_prev_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0,
                          desired_gpu_index : int = 0,
                          device: torch.device = torch.device('cuda'),
                          batch_size: int = 1,
                          apply_mask_to_prediction: bool = False):
    print("\n#######################################################################\nPlease cite the following paper "
          "when using nnU-Net:\n"
          "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
          "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
          "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    # 假設你想要在某個特定 GPU 上執行（例如GPU 1，編號從0開始）
    #desired_gpu_index = 0  # 修改此處來指定你希望使用的 GPU 編號

    # 檢查是否為 CUDA 設備，並指定 GPU 編號
    if device.type == 'cuda':
        device = torch.device(type='cuda', index=desired_gpu_index)  # 根據 desired_gpu_index 設定具體的 GPU

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    # let's store the input arguments so that its clear what was used to generate the prediction
    my_init_kwargs = {}
    for k in inspect.signature(predict_from_raw_data).parameters.keys():
        my_init_kwargs[k] = locals()[k]
    my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
    # safety precaution.
    recursive_fix_for_json_export(my_init_kwargs)
    maybe_mkdir_p(output_folder)
    save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    if use_folds is None:
        use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    # load all the stuff we need from the model_training_output_dir
    # 這邊獲得都是模型的參數
    parameters, configuration_manager, inference_allowed_mirroring_axes,     plans_manager, dataset_json, network, trainer_name =         load_what_we_need(model_training_output_dir, use_folds, checkpoint_name)
    
    print('總共有幾個網路parameters(同時拿幾個網路預測):', len(parameters)) #用來得知網路參數有幾個
    
    #這邊先不用到
    """
    # check if we need a prediction from the previous stage
    if configuration_manager.previous_stage_name is not None:
        if folder_with_segs_from_prev_stage is None:
            print(f'WARNING: The requested configuration is a cascaded model and requires predctions from the '
                  f'previous stage! folder_with_segs_from_prev_stage was not provided. Trying to run the '
                  f'inference of the previous stage...')
            folder_with_segs_from_prev_stage = join(output_folder,
                                                    f'prediction_{configuration_manager.previous_stage_name}')
            predict_from_raw_data(list_of_lists_or_source_folder,
                                  folder_with_segs_from_prev_stage,
                                  get_output_folder(plans_manager.dataset_name,
                                                    trainer_name,
                                                    plans_manager.plans_name,
                                                    configuration_manager.previous_stage_name),
                                  use_folds, tile_step_size, use_gaussian, use_mirroring, perform_everything_on_gpu,
                                  verbose, False, overwrite, checkpoint_name,
                                  num_processes_preprocessing, num_processes_segmentation_export, None,
                                  num_parts=num_parts, part_id=part_id, device=device)
    """

    # sort out input and output filenames
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])
        # Mask 檔名不含 _0000，用 case ID 對應
        if Mask_list_of_lists_or_Mask_folder is not None and isinstance(Mask_list_of_lists_or_Mask_folder, str):
            _mask_dir = Mask_list_of_lists_or_Mask_folder
            _ending = dataset_json['file_ending']
            Mask_list_of_lists_or_Mask_folder = []
            for case_files in list_of_lists_or_source_folder:
                # case_files[0] = .../CMB_001000_0000.nii.gz → case_id = CMB_001000
                _case_id = os.path.basename(case_files[0])[:-(len(_ending) + 5)]
                _mask_file = join(_mask_dir, _case_id + _ending)
                Mask_list_of_lists_or_Mask_folder.append(_mask_file if isfile(_mask_file) else None)
        elif Mask_list_of_lists_or_Mask_folder is None:
            Mask_list_of_lists_or_Mask_folder = [None] * len(list_of_lists_or_source_folder)
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    # Bug fix: mask list 同步切片，否則 image 跟 mask 對不上（image[i] 對 mask[i*num_parts+part_id]）
    if Mask_list_of_lists_or_Mask_folder is not None and isinstance(Mask_list_of_lists_or_Mask_folder, list):
        Mask_list_of_lists_or_Mask_folder = Mask_list_of_lists_or_Mask_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')
    print('list_of_lists_or_source_folder example:', list_of_lists_or_source_folder[0])
    print('Mask_list_of_lists_or_Mask_folder:', Mask_list_of_lists_or_Mask_folder[0] if Mask_list_of_lists_or_Mask_folder else 'None')

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
                                 folder_with_segs_from_prev_stage is not None else None for i in caseids]
    # remove already predicted files form the lists
    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
        # Bug fix: Mask_list_of_lists_or_Mask_folder 也必須同步過濾，否則 zip 對 image[i] 對到錯的 mask[i]（原索引）→ shape mismatch
        if Mask_list_of_lists_or_Mask_folder is not None and isinstance(Mask_list_of_lists_or_Mask_folder, list):
            Mask_list_of_lists_or_Mask_folder = [Mask_list_of_lists_or_Mask_folder[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')
        # caseids = [caseids[i] for i in not_existing_indices]

    # ⭐ Mask-fusion 架構（DeepConcat / SPADE 系列）：讓 mask channel resample 走 nearest neighbor
    # nnUNet 預設 resampling_fn_data 對所有 channel 都用 order=3 cubic，會把整數 label 弄成浮點
    # (如 vessel8 0-8 → 1.18, 3.6, 7.99...)。改用 per-channel 版：image ch 保持 cubic、mask ch 用 order=0
    _has_mask_channel = (
        hasattr(network, 'image_channels')
        and hasattr(network, 'mask_classes')
    )
    if _has_mask_channel:
        _mask_ch_idx = int(network.image_channels)  # 最後一個 channel index (image channels 之後就是 mask)
        from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape as _rs

        def _resample_fn_per_channel(data, new_shape, current_spacing, new_spacing):
            # data shape: (C, D, H, W)。前 _mask_ch_idx 個是 image channel，最後 1 個是 mask channel
            _img = data[:_mask_ch_idx]
            _mask = data[_mask_ch_idx:_mask_ch_idx + 1]
            _img_r = _rs(_img, new_shape, current_spacing, new_spacing,
                         is_seg=False, order=3, order_z=0, force_separate_z=None)
            _mask_r = _rs(_mask, new_shape, current_spacing, new_spacing,
                          is_seg=True, order=1, order_z=0, force_separate_z=None)
            return np.concatenate([_img_r, _mask_r], axis=0)

        # ConfigurationManager 的 resampling_fn_data 是 @property @lru_cache — 不能直接 set
        # 動態 subclass 這個 instance，用 override property 蓋掉，只影響這個 instance
        _CM = type(configuration_manager)
        _cached_fn = _resample_fn_per_channel  # closure 綁定
        _MaskCM = type(
            'MaskChannelAware' + _CM.__name__, (_CM,),
            {'resampling_fn_data': property(lambda _self: _cached_fn)}
        )
        configuration_manager.__class__ = _MaskCM
        print(f'[mask-channel] override resampling_fn_data: image ch [0:{_mask_ch_idx}] cubic, mask ch [{_mask_ch_idx}] nearest')

    # placing this into a separate function doesnt make sense because it needs so many input variables...
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    #print('seg_from_prev_stage_files:', seg_from_prev_stage_files) #這邊原本都是None
    
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, Mask_list_of_lists_or_Mask_folder, preprocessor,
                            output_filename_truncated, plans_manager, dataset_json,
                            configuration_manager, num_processes)
    # MultiThreadedAugmenter spawns a worker even at num_processes=1, and the
    # server sets the start method to spawn (fork inherits a CUDA context and
    # deadlocks on first use). Spawning pickles the configuration manager --
    # which, for the mask-fusion architectures, is the subclass built with
    # type() a few lines above and therefore has no importable name. The result
    # is a PicklingError before any inference happens.
    #
    # For one case there is no work to parallelise, so take the in-process path
    # instead of making the dynamic class picklable. Same fix, same reason, as
    # brain-parcellation c2399f7 on the aneurysm predictor.
    if num_processes <= 1:
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        mta = SingleThreadedAugmenter(ppa, NumpyToTensor())
    else:
        mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda')
    
    # precompute gaussian
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)).half()
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to(device)
    print('inference_gaussian.shape:', inference_gaussian.shape)

    # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
    label_manager = plans_manager.get_label_manager(dataset_json)

    # 自動偵測 activation：region-based labels (list) → sigmoid, 否則 softmax
    _has_regions = any(isinstance(v, list) for v in dataset_json.get('labels', {}).values())
    _use_sigmoid = _has_regions
    print(f'[MUTP] labels has_regions={_has_regions} → activation={"sigmoid" if _use_sigmoid else "softmax"}')
    num_seg_heads = label_manager.num_segmentation_heads
    #num_seg_heads 這邊為 0背景 1.動脈瘤，所以為2
    #print('num_seg_heads:', num_seg_heads)

    # go go go
    # spawn allows the use of GPU in the background process in case somebody wants to do this. Not recommended. Trust me.
    # export_pool = multiprocessing.get_context('spawn').Pool(num_processes_segmentation_export)
    # export_pool = multiprocessing.Pool(num_processes_segmentation_export)
    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        network = network.to(device)

        r = []
        with torch.no_grad():
            for preprocessed, nii_path in zip(mta, list_of_lists_or_source_folder):
                start_time = time.time()
                data = preprocessed['data']
                data_vessel = preprocessed['seg']
                #print('data:', data.shape, 'data_vessel:', data_vessel.shape)
                #讀取nifti只是為了affine
                img_nii = nib.load(str(nii_path[0]))
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)
                
                if isinstance(data_vessel, str):
                    data_vessel = torch.from_numpy(np.load(data_vessel))

                # 沒提供 mask（沒 inference_mask / mask_dir）時 data_vessel 是空/全 0/全 -1
                # nnU-Net preprocessor 對無 seg 的 case 預設用 -1 sentinel 填充 → max <= 0
                # → sliding window 過濾條件 `> 0` 永遠失敗 → 全 patch 跳過 → 全 0 prediction
                # Fix：把這 3 種情況都視為「無限制」，用 ones 取代讓所有 patch 都被預測
                _t_v = data_vessel
                _no_mask = (_t_v is None)
                if not _no_mask:
                    try:
                        _no_mask = (float(_t_v.max()) <= 0)  # 全 0 或全 -1（sentinel）都算
                    except Exception:
                        _no_mask = False
                if _no_mask:
                    if isinstance(data, torch.Tensor):
                        data_vessel = torch.ones_like(data[:1] if data.ndim == 4 else data)
                    else:
                        data_vessel = np.ones_like(data[:1] if data.ndim == 4 else data)

                ofile = preprocessed['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:')
                print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')
                print('configuration_manager.patch_size:', configuration_manager.patch_size)
                
                properties = preprocessed['data_properites'] #組回nifti的參數

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))
                while not proceed:
                    sleep(1)
                    proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))

                # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
                # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
                # things a lot faster for some datasets.
                prediction = None
                classification_prob = None
                overwrite_perform_everything_on_gpu = perform_everything_on_gpu
                #目前是走perform_everything_on_gpu = 1
                if perform_everything_on_gpu:
                    try:
                        for params in parameters:
                            network.load_state_dict(params)
                            if prediction is None:
                                prediction, classification_prob = predict_sliding_window_return_logits(
                            network, data, data_vessel, num_seg_heads,
                            configuration_manager.patch_size,
                            mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                            tile_step_size=tile_step_size,
                            use_gaussian=use_gaussian,
                            precomputed_gaussian=inference_gaussian,
                            perform_everything_on_gpu=perform_everything_on_gpu,
                            verbose=verbose,
                            device=device,
                            batch_size=batch_size, use_sigmoid=_use_sigmoid,
                            apply_mask_to_prediction=apply_mask_to_prediction)
                            else:
                                pred_tmp, cls_tmp = predict_sliding_window_return_logits(
                                    network, data, data_vessel, num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device,
                                    batch_size=batch_size, use_sigmoid=_use_sigmoid,
                            apply_mask_to_prediction=apply_mask_to_prediction)
                                prediction += pred_tmp
                                classification_prob += cls_tmp
                            if len(parameters) > 1:
                                prediction /= len(parameters)
                                classification_prob /= len(parameters)

                    except RuntimeError:
                        print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                              'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                        print('Error:')
                        traceback.print_exc()
                        prediction = None
                        classification_prob = None
                        overwrite_perform_everything_on_gpu = False

                #如果gpu失敗，走以下
                if prediction is None:
                    for params in parameters:
                        network.load_state_dict(params)
                        if prediction is None:
                            prediction, classification_prob = predict_sliding_window_return_logits(
                                network, data, data_vessel, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                                batch_size=batch_size, use_sigmoid=_use_sigmoid,
                            apply_mask_to_prediction=apply_mask_to_prediction)
                        else:
                            pred_tmp, cls_tmp = predict_sliding_window_return_logits(
                                network, data, data_vessel, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                                batch_size=batch_size, use_sigmoid=_use_sigmoid,
                            apply_mask_to_prediction=apply_mask_to_prediction)
                            prediction += pred_tmp
                            classification_prob += cls_tmp
                        if len(parameters) > 1:
                            prediction /= len(parameters)
                            classification_prob /= len(parameters)

                print('Prediction done, transferring to CPU if needed')
                prediction = prediction.to('cpu').numpy()
                classification_prob = classification_prob.to('cpu').numpy()
                
                #print('final prediction.shape:', prediction.shape)
                #print('final classification_prob.shape:', classification_prob.shape)
                if should_i_save_to_file(prediction, r, export_pool):
                    print(
                        'output is either too large for python process-process communication or all export workers are '
                        'busy. Saving temporarily to file...')
                    np.save(ofile + '.npy', prediction)
                    prediction = ofile + '.npy'
                    np.save(ofile + '_cls.npy', classification_prob)
                    classification_prob = ofile + '_cls.npy'

                """
                # this needs to go into background processes
                # export_prediction(prediction, properties, configuration_name, plans, dataset_json, ofile,
                #                   save_probabilities)
                print('sending off prediction to background worker for resampling and export')
                r.append(
                    export_pool.starmap_async(
                        export_prediction_probabilities, ((prediction, properties, configuration_manager, plans_manager,
                                                          dataset_json, ofile, save_probabilities),)
                    )
                )
                print(f'done with {os.path.basename(ofile)}')
                """
                print(f"[Done] spend {time.time() - start_time:.2f} sec")
                export_prediction_probabilities(prediction, classification_prob, properties, data_vessel, img_nii, configuration_manager, plans_manager,
                                                dataset_json, ofile, save_probabilities)
                
                print(f"[Done] spend {time.time() - start_time:.2f} sec")
        #[i.get() for i in r]

    # we need these two if we want to do things with the predictions like for example apply postprocessing
    shutil.copy(join(model_training_output_dir, 'dataset.json'), join(output_folder, 'dataset.json'))
    shutil.copy(join(model_training_output_dir, 'plans.json'), join(output_folder, 'plans.json'))


# predict_entry_point 和 __main__ 已移除 — 由 MUTP CLI 呼叫 predict_from_raw_data()

