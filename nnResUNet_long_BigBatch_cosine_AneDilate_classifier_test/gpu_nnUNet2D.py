# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:44:17 2023

nnU-Net簡易版inference pipeline (2D版本)

@author: user
"""

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
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json, save_pickle
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_softmax
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

import warnings
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter
from torch import nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import torch.nn.functional as F

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel as nib

from copy import deepcopy
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from multiprocessing import Pool

#這邊是定義DataLoader
class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]], list_of_segs_from_prev_stage_files: Union[List[None], List[str]],
                 preprocessor: DefaultPreprocessor, output_filenames_truncated: List[str],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1):
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json = \
            preprocessor, plans_manager, configuration_manager, dataset_json

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
        
        data, seg, data_properites = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)

        if np.prod(data.shape) > (2e9 / 4 * 0.85):
            np.save(ofile + '.npy', data)
            data = ofile + '.npy'

        return {'data': data, 'seg': seg, 'data_properites': data_properites, 'ofile': ofile}

#讀取需要的資訊
def load_what_we_need(model_training_output_dir, use_folds, checkpoint_name, plans_json_name='nnUNetPlans.json'):
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, plans_json_name))
    
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
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision=False)
    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network, trainer_name

#讀資料夾順序
def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
    print('use_folds is None, attempting to auto detect available folds')
    fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
    fold_folders = [i for i in fold_folders if i != 'fold_all']
    fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
    use_folds = [int(i.split('_')[-1]) for i in fold_folders]
    print(f'found the following folds: {use_folds}')
    return use_folds

#計算高斯map (2D版本)
def compute_gaussian(tile_size: Tuple[int, ...], sigma_scale: float = 1. / 8, dtype=np.float16) \
        -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

#計算 patch box (2D版本)
def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)

    return steps

#做sliding_window生成器 (2D版本 - 逐slice處理)
def get_sliding_window_generator_2d(image_size: Tuple[int, int, int], tile_size: Tuple[int, int], tile_step_size: float,
                                    verbose: bool = False):
    """
    2D sliding window generator for (C, H, W) images
    image_size: (depth, height, width) - 實際上是 (slices, H, W)
    tile_size: (height, width) - 2D patch size
    """
    num_slices = image_size[0]
    steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
    
    if verbose: 
        print(f'n_steps {num_slices * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
              f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
    
    # 對每個 slice 進行 2D sliding window
    for d in range(num_slices):
        for sx in steps[0]:
            for sy in steps[1]:
                # slicer 格式: [channel, slice, height, width]
                slicer = tuple([slice(None), d, slice(sx, sx + tile_size[0]), slice(sy, sy + tile_size[1])])
                yield slicer

#是否要更複雜的inference(可選)
def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None, 
                            has_classifier_output: bool = False) \
        -> torch.Tensor:
    if has_classifier_output:
        prediction, cls = network(x)
    else:
        prediction = network(x)

    if mirror_axes is not None:
        # 2D 的 mirror_axes: 0=height, 1=width
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        prediction /= num_predictons
    return prediction

#sliding_window的pipeline (2D版本)
def predict_sliding_window_return_logits_2d(network: nn.Module,
                                            input_image: Union[np.ndarray, torch.Tensor],
                                            vessel_image: Union[np.ndarray, torch.Tensor],
                                            num_segmentation_heads: int,
                                            tile_size: Tuple[int, int],
                                            mirror_axes: Tuple[int, ...] = None,
                                            tile_step_size: float = 0.5,
                                            use_gaussian: bool = True,
                                            precomputed_gaussian: torch.Tensor = None,
                                            perform_everything_on_gpu: bool = True,
                                            verbose: bool = True,
                                            device: torch.device = torch.device('cuda'),
                                            batch_size: int = 1,
                                            has_classifier_output: bool = False) -> Union[np.ndarray, torch.Tensor]:
    if perform_everything_on_gpu:
        assert device.type == 'cuda', 'Can use perform_everything_on_gpu=True only when device="cuda"'

    network = network.to(device)
    network.eval()

    empty_cache(device)
    
    with torch.no_grad():
        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            # 2D: input_image shape 應該是 (c, slices, h, w)
            assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, slices, h, w)'

            if not torch.cuda.is_available():
                if perform_everything_on_gpu:
                    print('WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...')
                perform_everything_on_gpu = False

            results_device = device if perform_everything_on_gpu else torch.device('cpu')

            if verbose: print("step_size:", tile_step_size)
            if verbose: print("mirror_axes:", mirror_axes)

            if not isinstance(input_image, torch.Tensor):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_image = torch.from_numpy(input_image)

            # 對 2D patch，只需要在 H, W 維度 padding
            data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True, None)
            
            # Vessel image padding - 對齊到 data 的大小
            pad_height = (data.shape[2] - vessel_image.shape[2])
            pad_width = (data.shape[3] - vessel_image.shape[3])
            
            # 2D padding 順序: (左, 右, 上, 下)
            padding = (pad_width // 2, pad_width - pad_width // 2,
                      pad_height // 2, pad_height - pad_height // 2)
            
            data_vessel = F.pad(vessel_image, padding, mode='constant', value=0)
                        
            print("step_size:", tile_step_size)
            print("mirror_axes:", mirror_axes)
            print('data pad後的大小:', data.shape)
            print('data_vessel pad後的大小:', data_vessel.shape)

            if use_gaussian:
                gaussian = torch.from_numpy(
                    compute_gaussian(tile_size, sigma_scale=1. / 8)) if precomputed_gaussian is None else precomputed_gaussian
                gaussian = gaussian.half()
                mn = gaussian.min()
                if mn == 0:
                    gaussian.clip_(min=mn)
            else:
                gaussian = None
                    
            # 使用 2D sliding window generator
            slicers = get_sliding_window_generator_2d(data.shape[1:], tile_size, tile_step_size, verbose=verbose)

            # Preallocate results
            try:
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                if use_gaussian and gaussian is not None:
                    gaussian = gaussian.to(results_device)
            except RuntimeError:
                results_device = torch.device('cpu')
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                if use_gaussian and gaussian is not None:
                    gaussian = gaussian.to(results_device)
            finally:
                empty_cache(device)

            if use_gaussian:
                patches_to_process = []
                slicers_to_process = []
                
                for sl in slicers:
                    if torch.sum(data_vessel[sl]) > 0:
                        workon = data[sl][None]
                        patches_to_process.append(workon)
                        slicers_to_process.append(sl)
                    else:
                        prediction = torch.zeros((num_segmentation_heads, *tile_size)).to(results_device)
                        predicted_logits[sl] += prediction * gaussian
                        n_predictions[sl[1:]] += gaussian
                
                if len(patches_to_process) > 0:
                    if verbose:
                        print(f"[Gaussian模式] 處理 {len(patches_to_process)} 個有血管的 patches，使用 batch_size={batch_size}")
                    
                    for i in range(0, len(patches_to_process), batch_size):
                        batch_end = min(i + batch_size, len(patches_to_process))
                        batch_patches = patches_to_process[i:batch_end]
                        batch_slicers = slicers_to_process[i:batch_end]
                        
                        batch_tensor = torch.cat(batch_patches, dim=0).to(device, non_blocking=False)
                        batch_predictions = maybe_mirror_and_predict(network, batch_tensor, mirror_axes, has_classifier_output).to(results_device)
                        
                        for j, (prediction, sl) in enumerate(zip(batch_predictions, batch_slicers)):
                            prediction = torch.softmax(prediction, 0)
                            predicted_logits[sl] += prediction * gaussian
                            n_predictions[sl[1:]] += gaussian
            else:
                patches_to_process = []
                slicers_to_process = []
                all_slicers = list(slicers)
                
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
                        
                        batch_tensor = torch.cat(batch_patches, dim=0).to(device, non_blocking=False)
                        
                        start_time_batch = time.time()
                        batch_predictions = maybe_mirror_and_predict(network, batch_tensor, mirror_axes, has_classifier_output).to(results_device)
                        print(f"[Done] maybe_mirror_and_predict no. {i} spend {time.time() - start_time_batch:.3f} sec")
                        
                        for j, (prediction, sl) in enumerate(zip(batch_predictions, batch_slicers)):
                            prediction = torch.softmax(prediction, 0)
                            predicted_logits[sl] += prediction
                            n_predictions[sl[1:]] += 1
                
                if len(patches_to_process) < len(all_slicers):
                    background_prediction = torch.zeros((num_segmentation_heads, *tile_size), device=results_device)
                    background_prediction[0] = 1.0
                    
                    for sl in all_slicers:
                        if torch.sum(data_vessel[sl]) == 0:
                            predicted_logits[sl] += background_prediction
                            n_predictions[sl[1:]] += 1

            # 安全除法
            mask = n_predictions > 0
            predicted_logits = torch.where(mask.unsqueeze(0), 
                                         predicted_logits / n_predictions.unsqueeze(0), 
                                         predicted_logits)
            
            if verbose:
                zero_predictions = torch.sum(n_predictions == 0).item()
                total_voxels = torch.numel(n_predictions)
                if zero_predictions > 0:
                    print(f"警告：有 {zero_predictions}/{total_voxels} 個體素沒有被任何 patch 覆蓋到")
            
            # 與vessel相乘
            repeat_vessel = data_vessel.repeat(num_segmentation_heads, 1, 1, 1)
            predicted_logits = predicted_logits * repeat_vessel.to(results_device)

    empty_cache(device)
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

#存出probabilities map
def write_probabilities(seg, output_fname, img_nii):
    seg = seg.transpose((2, 1, 0)).astype(np.float32)
    
    affine = img_nii.affine
    header = img_nii.header.copy()
    new_nii = nib.nifti1.Nifti1Image(seg, affine, header=header)
    
    nib.save(new_nii, output_fname)

#輸出結果
def export_prediction_probabilities(predicted_array_or_file: Union[np.ndarray, str], properties_dict: dict,
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

    predicted_array_or_file = predicted_array_or_file.astype(np.float32)
    print('before')
    print('predicted_array_or_file.shape:', predicted_array_or_file.shape)
    print('np.max(predicted_array_or_file):', np.max(predicted_array_or_file))
    print('np.median(predicted_array_or_file):', np.median(predicted_array_or_file))

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    
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
    
    probs_reverted_cropping = label_manager.revert_cropping(predicted_array_or_file,
                                                            properties_dict['bbox_used_for_cropping'],
                                                            properties_dict['shape_before_cropping'])
    
    print('revert cropping')
    print('probs_reverted_cropping.shape:', probs_reverted_cropping.shape)
    print('np.max(probs_reverted_cropping):', np.max(probs_reverted_cropping))
    print('np.median(probs_reverted_cropping):', np.median(probs_reverted_cropping))
    
    probs_reverted_cropping = np.expand_dims(probs_reverted_cropping[1,:,:,:], axis=0)
    
    if probs_reverted_cropping is None:
        raise ValueError("Reverting cropping failed, 'probs_reverted_cropping' is None.")
        
    probs_reverted_cropping = probs_reverted_cropping.transpose([0] + [i + 1 for i in
                                                                plans_manager.transpose_backward])
    
    write_probabilities(probs_reverted_cropping[0,:,:,:], output_file_truncated + dataset_json_dict_or_file['file_ending'], img_nii)

#從raw data開始處理的pipeline
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
                          plans_json_name: str = 'nnUNetPlans.json',
                          has_classifier_output: bool = False,
                          num_processes_preprocessing: int = default_num_processes,
                          num_processes_segmentation_export: int = default_num_processes,
                          folder_with_segs_from_prev_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0,
                          desired_gpu_index : int = 0,
                          device: torch.device = torch.device('cuda'),
                          batch_size: int = 1):
    print("\n#######################################################################\nPlease cite the following paper "
          "when using nnU-Net:\n"
          "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
          "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
          "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    if device.type == 'cuda':
        device = torch.device(type='cuda', index=desired_gpu_index)

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    my_init_kwargs = {}
    for k in inspect.signature(predict_from_raw_data).parameters.keys():
        my_init_kwargs[k] = locals()[k]
    my_init_kwargs = deepcopy(my_init_kwargs)
    recursive_fix_for_json_export(my_init_kwargs)
    maybe_mkdir_p(output_folder)
    save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    if use_folds is None:
        use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_what_we_need(model_training_output_dir, use_folds, checkpoint_name, plans_json_name)
    
    print('總共有幾個網路parameters(同時拿幾個網路預測):', len(parameters))

    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])
        Mask_list_of_lists_or_Mask_folder = create_lists_from_splitted_dataset_folder(Mask_list_of_lists_or_Mask_folder,
                                                                                   dataset_json['file_ending'])
        # 修正：從 List[List[str]] 提取成 List[str]，因為 vessel mask 只有一個 _0000.nii.gz 檔案
        Mask_list_of_lists_or_Mask_folder = [mask_files[0] if mask_files else None 
                                              for mask_files in Mask_list_of_lists_or_Mask_folder]
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    Mask_list_of_lists_or_Mask_folder = Mask_list_of_lists_or_Mask_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')
    print('list_of_lists_or_source_folder example:', list_of_lists_or_source_folder[0])
    print('Mask_list_of_lists_or_Mask_folder (vessel mask) example:', Mask_list_of_lists_or_Mask_folder[0])
    print(f'  → Image has {len(list_of_lists_or_source_folder[0])} series/channels')
    print(f'  → Vessel mask is a single file: {isinstance(Mask_list_of_lists_or_Mask_folder[0], str)}')

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
                                 folder_with_segs_from_prev_stage is not None else None for i in caseids]
    
    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        Mask_list_of_lists_or_Mask_folder = [Mask_list_of_lists_or_Mask_folder[i] for i in not_existing_indices]
        seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')

    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, Mask_list_of_lists_or_Mask_folder, preprocessor,
                            output_filename_truncated, plans_manager, dataset_json,
                            configuration_manager, num_processes)
    mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda')
    
    # 2D patch size
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)).half()
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to(device)
    print('inference_gaussian.shape:', inference_gaussian.shape)

    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        network = network.to(device)

        r = []
        with torch.no_grad():
            for preprocessed, nii_path in zip(mta, list_of_lists_or_source_folder):
                start_time = time.time()
                data = preprocessed['data']
                data_vessel = preprocessed['seg']
                
                img_nii = nib.load(str(nii_path[0]))
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)
                
                if isinstance(data_vessel, str):
                    data_vessel = torch.from_numpy(np.load(data_vessel))

                ofile = preprocessed['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:')
                print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')
                print('configuration_manager.patch_size:', configuration_manager.patch_size)
                
                properties = preprocessed['data_properites']

                proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))
                while not proceed:
                    sleep(1)
                    proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))

                prediction = None
                overwrite_perform_everything_on_gpu = perform_everything_on_gpu
                
                if perform_everything_on_gpu:
                    try:
                        for params in parameters:
                            network.load_state_dict(params)
                            if prediction is None:
                                prediction = predict_sliding_window_return_logits_2d(
                            network, data, data_vessel, num_seg_heads,
                            configuration_manager.patch_size,
                            mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                            tile_step_size=tile_step_size,
                            use_gaussian=use_gaussian,
                            precomputed_gaussian=inference_gaussian,
                            perform_everything_on_gpu=perform_everything_on_gpu,
                            verbose=verbose,
                            device=device,
                            batch_size=batch_size,
                            has_classifier_output=has_classifier_output)
                            else:
                                prediction += predict_sliding_window_return_logits_2d(
                                    network, data, data_vessel, num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device,
                                    batch_size=batch_size,
                                    has_classifier_output=has_classifier_output)
                            if len(parameters) > 1:
                                prediction /= len(parameters)

                    except RuntimeError:
                        print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                              'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                        print('Error:')
                        traceback.print_exc()
                        prediction = None
                        overwrite_perform_everything_on_gpu = False

                if prediction is None:
                    for params in parameters:
                        network.load_state_dict(params)
                        if prediction is None:
                            prediction = predict_sliding_window_return_logits_2d(
                                network, data, data_vessel, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                                batch_size=batch_size,
                                has_classifier_output=has_classifier_output)
                        else:
                            prediction += predict_sliding_window_return_logits_2d(
                                network, data, data_vessel, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                                batch_size=batch_size,
                                has_classifier_output=has_classifier_output)
                        if len(parameters) > 1:
                            prediction /= len(parameters)

                print('Prediction done, transferring to CPU if needed')
                prediction = prediction.to('cpu').numpy()
                
                if should_i_save_to_file(prediction, r, export_pool):
                    print(
                        'output is either too large for python process-process communication or all export workers are '
                        'busy. Saving temporarily to file...')
                    np.save(ofile + '.npy', prediction)
                    prediction = ofile + '.npy'

                print(f"[Done] spend {time.time() - start_time:.2f} sec")
                export_prediction_probabilities(prediction, properties, data_vessel, img_nii, configuration_manager, plans_manager,
                                                dataset_json, ofile, save_probabilities)
                
                print(f"[Done] spend {time.time() - start_time:.2f} sec")


#主程式
if __name__ == "__main__":
    from multiprocessing import Pool
    predict_from_raw_data('/data/chuan/nnUNet/nnUNet_raw/Dataset040_DeepInfarct/Normalized_Image_Test/',
                          '/data/chuan/nnUNet/nnUNet_raw/Dataset040_DeepInfarct/Vessel_Test/',
                          '/data/chuan/nnUNet/nnUNet_inference/Dataset040_DeepInfarct/2d/Test_gaussian1',
                          '/data/chuan/nnUNet/nnUNet_results/Dataset040_DeepInfarct/nnUNetTrainer__nnUNetPlans__2d',
                          (0,),
                          0.25,
                          use_gaussian=True,
                          use_mirroring=False,
                          perform_everything_on_gpu=True,
                          verbose=True,
                          save_probabilities=False,
                          overwrite=False,
                          checkpoint_name='checkpoint_best.pth',
                          plans_json_name='nnUNetPlans.json',
                          has_classifier_output=False,
                          num_processes_preprocessing=2,
                          num_processes_segmentation_export=3,
                          desired_gpu_index = 0,
                          batch_size=64  # 2D 可以使用更大的 batch_size
                          )

