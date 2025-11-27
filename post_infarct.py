# -*- coding: utf-8 -*-
"""
Infarct 後處理模組
處理 nnUNet 預測結果，生成視覺化圖片和報告

@author: chuan
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure
import json
from collections import OrderedDict
import numba as nb
from nii_transforms import nii_img_replace


@nb.jit(nopython=True, nogil=True)
def parcellation(nii_array, x1, x2, c1, new_array):
    """
    使用最近鄰方法重新分配腦區標籤
    
    參數:
        nii_array: 原始分割陣列
        x1: 需要重新分配的座標
        x2: 參考座標
        c1: 標籤類別
        new_array: 輸出陣列
    """
    y_coord = x1[0]
    x_coord = x1[1]
    for y_i, x_i in zip(y_coord, x_coord):
        max_v = nii_array.shape[0]*nii_array.shape[0] + nii_array.shape[1]*nii_array.shape[1]
        for class_idx in range(c1.shape[0]):
            new_x2 = (x2[class_idx,0,:] - y_i)*(x2[class_idx,0,:] - y_i) + (x2[class_idx,1,:] - x_i)*(x2[class_idx,1,:] - x_i)
            for k in new_x2:
                if max_v > k:
                    max_v = k
                    new_array[y_i, x_i] = c1[class_idx]
    return new_array


def data_translate(img, nii):
    """將影像轉換為模型預測所需格式"""
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    header = nii.header.copy()
    pixdim = header['pixdim']
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    return img


def data_translate_back(img, nii):
    """將影像轉換回原始格式"""
    header = nii.header.copy()
    pixdim = header['pixdim']
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    return img


def create_json_report(json_path, patient_id, volume, mean_adc, report):
    """
    創建 JSON 報告
    
    參數:
        json_path: JSON 檔案路徑
        patient_id: 病人 ID
        volume: 梗塞體積 (ml)
        mean_adc: 平均 ADC 值
        report: 報告文字
    """
    json_dict = OrderedDict()
    json_dict["PatientID"] = patient_id
    json_dict["volume"] = volume
    json_dict["MeanADC"] = mean_adc
    json_dict["Report"] = report
    
    with open(json_path, 'w', encoding='utf8') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, 
                 separators=(',', ': '), ensure_ascii=False)


def postprocess_infarct_results(
    patient_id,
    path_nnunet,
    path_code,
    path_output,
    path_json,
):
    """
    處理 Infarct 預測結果的後處理流程
    
    參數:
        patient_id: 病人 ID
        path_nnunet: nnUNet 資料夾路徑
        path_code: 程式碼根目錄路徑
        path_output: 最終輸出資料夾路徑
        path_json: JSON 輸出路徑
    
    資料夾結構:
        - 讀取: nnUNet/Image_nii/ (ADC.nii.gz, DWI1000.nii.gz, SynthSEG.nii.gz)
        - 讀取: nnUNet/ (Pred.nii.gz - 由 gpu_infarct.py 生成)
        - 輸出: nnUNet/Result/ (處理後的 Pred.nii.gz, PNG 圖片)
        - 輸出: path_output (最終結果檔案)
    
    返回:
        bool: 處理是否成功
    """
    ROWS = 256
    COLS = 256
    
    # 定義路徑
    path_nii = os.path.join(path_nnunet, 'Image_nii')  # 直接從 nnUNet/Image_nii 讀取
    path_result = os.path.join(path_nnunet, 'Result')
    path_dcm = os.path.join(path_nnunet, 'Dicom')
    
    # 建立必要資料夾（簡化結構）
    os.makedirs(path_result, exist_ok=True)
    
    # 檢查必要檔案是否存在
    required_files = {
        'ADC': os.path.join(path_nii, 'ADC.nii.gz'),
        'DWI1000': os.path.join(path_nii, 'DWI1000.nii.gz'),
        'SynthSEG': os.path.join(path_nii, 'SynthSEG.nii.gz'),
        'prediction': os.path.join(path_nii, 'Pred.nii.gz')  # 直接從 nnUNet 根目錄讀取
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"錯誤：找不到必要檔案 {name}: {path}")
            return False
    
    # 載入影像資料（直接從 nii.gz 載入）
    ADC_nii = nib.load(required_files['ADC'])
    ADC_array = np.array(ADC_nii.dataobj)
    DWI1000_nii = nib.load(required_files['DWI1000'])
    DWI1000_array = np.array(DWI1000_nii.dataobj)
    SynthSEG_nii = nib.load(required_files['SynthSEG'])
    SynthSEG_array = np.array(SynthSEG_nii.dataobj)
    
    # 載入預測結果
    Y_pred_nii = nib.load(required_files['prediction'])
    Y_pred = np.array(Y_pred_nii.dataobj)
    
    # 計算體素大小
    header_true = DWI1000_nii.header.copy()
    pixdim = header_true['pixdim']
    ml_size = (pixdim[1] * pixdim[2] * pixdim[3]) / 1000
    
    # 轉換影像格式
    ADC_array = data_translate(ADC_array, ADC_nii)
    DWI1000_array = data_translate(DWI1000_array, DWI1000_nii)
    SynthSEG_array = data_translate(SynthSEG_array, SynthSEG_nii)
    Y_pred = data_translate(Y_pred, DWI1000_nii)

    #DWI1000_nor = (DWI - np.mean(DWI)) / np.std(DWI)
    DWI1000_nor = (DWI1000_array - np.mean(DWI1000_array)) / np.std(DWI1000_array)
    
    y_i, x_i, z_i = ADC_array.shape
    
    # 二值化預測結果
    y_pred_nor1 = Y_pred.astype('int').copy()
    
    # 計算不同版本的預測結果
    Pred_ADCth = (y_pred_nor1 * ADC_array).copy()
    Pred_SynthSEG = (y_pred_nor1 * SynthSEG_array).copy()
    
    # 儲存 NIfTI 結果到 Result 資料夾（path_result）
    new_y_pred_nor1 = data_translate_back(y_pred_nor1, DWI1000_nii).astype(int)
    new_y_pred_nor1_nii = nii_img_replace(DWI1000_nii, new_y_pred_nor1)
    nib.save(new_y_pred_nor1_nii, os.path.join(path_result, 'Pred.nii.gz'))
    nib.save(new_y_pred_nor1_nii, os.path.join(path_output, 'Pred_Infarct.nii.gz'))
    
    Pred_ADCth_back = data_translate_back(Pred_ADCth, DWI1000_nii).astype(int)
    Pred_ADCth_nii = nii_img_replace(DWI1000_nii, Pred_ADCth_back)
    nib.save(Pred_ADCth_nii, os.path.join(path_result, 'Pred_ADCth.nii.gz'))
    nib.save(Pred_ADCth_nii, os.path.join(path_output, 'Pred_Infarct_ADCth.nii.gz'))
    
    Pred_SynthSEG_back = data_translate_back(Pred_SynthSEG, DWI1000_nii).astype(int)
    Pred_SynthSEG_nii = nii_img_replace(DWI1000_nii, Pred_SynthSEG_back)
    nib.save(Pred_SynthSEG_nii, os.path.join(path_result, 'Pred_synthseg.nii.gz'))
    nib.save(Pred_SynthSEG_nii, os.path.join(path_output, 'Pred_Infarct_synthseg.nii.gz'))
    
    # 聚類分析並移除小於閾值的區域
    y_pred_cluster = measure.label(y_pred_nor1, connectivity=3)
    cluster_num = np.max(y_pred_cluster)
    ml_voxel = 1 / ml_size
    
    for cluster_i in range(cluster_num):
        y_cl, x_cl, z_cl = np.where(y_pred_cluster == (cluster_i + 1))
        if len(y_cl) < 0.3 * ml_voxel or np.mean(ADC_array[y_cl, x_cl, z_cl]) > 800:
            y_pred_cluster[y_cl, x_cl, z_cl] = 0
    
    y_pred_cluster[y_pred_cluster > 0] = 1
    
    # 計算統計值
    total_mean_ADC = np.sum(ADC_array * y_pred_cluster)
    if np.sum(y_pred_cluster) > 0:
        MeanADC = int(total_mean_ADC / np.sum(y_pred_cluster))
    else:
        MeanADC = 0
    
    InfarctVoxel = int(np.sum(y_pred_cluster))
    InfarctVolume = round(InfarctVoxel * ml_size, 1)
    
    # 載入 SynthSEG 標籤配置
    json_file = os.path.join(path_code, 'dataset.json')
    with open(json_file) as f:
        json_data = json.load(f)
    
    Infarct_labels = json_data['Infarct_labels_OHIF']
    colorbar = json_data['Infarct_colors_OHIF']
    
    # 處理 SynthSEG 重新分配
    outside_infarct_labels = {}
    outside_labels = []
    for idx_labels, label in enumerate(Infarct_labels):
        if label != 'Background' and label != 'CSF':
            outside_infarct_labels[label] = Infarct_labels[label]
            for jdx_labels, seg_id in enumerate(Infarct_labels[label]):
                outside_labels.append(seg_id)
    
    outside_labels = np.array(outside_labels)
    
    # 重新分配 CSF 和背景區域
    BET_SynthSeg_array = SynthSEG_array.copy()
    BET_SynthSeg_array[BET_SynthSeg_array == Infarct_labels['Background']] = 0
    BET_SynthSeg_array[BET_SynthSeg_array == Infarct_labels['CSF']] = 0
    label_overlap = y_pred_cluster.copy() * BET_SynthSeg_array.copy()
    
    SynthSEG_mirror = (SynthSEG_array == Infarct_labels['CSF']) + (SynthSEG_array == Infarct_labels['Background'])
    label_outside = SynthSEG_mirror * y_pred_cluster.copy()
    outside_array = np.zeros(SynthSEG_array.shape)
    
    # 2D 切片處理重新分配
    for i in range(z_i):
        if np.sum(label_outside[:, :, i]) > 0 and np.sum(BET_SynthSeg_array[:, :, i]) > 0:
            label_slice = label_outside[:, :, i]
            SynthSeg_slice = BET_SynthSeg_array[:, :, i]
            new_slice = np.zeros((SynthSeg_slice.shape))
            intersection = np.intersect1d(outside_labels, np.unique(SynthSeg_slice))
            coordinate = np.array(np.where(label_slice == 1))
            long_class = max([len(np.where(SynthSeg_slice == y)[0]) for y in intersection])
            coordinate_class = np.zeros((len(intersection), 2, long_class))
            
            for i_c in range(intersection.shape[0]):
                c_coord = np.array(np.where(SynthSeg_slice == intersection[i_c]))
                coordinate_class[i_c, :, :c_coord.shape[1]] = c_coord
                coordinate_class[i_c, 0, c_coord.shape[1]+1:] = c_coord[0, 0]
                coordinate_class[i_c, 1, c_coord.shape[1]+1:] = c_coord[1, 0]
            
            new_slice = parcellation(SynthSeg_slice, coordinate, coordinate_class, intersection, new_slice)
            outside_array[:, :, i] = new_slice
    
    new_SynthSeg_array = label_overlap + outside_array
    
    # 計算每個腦區的梗塞體積
    new_Infarct_labels = {}
    total_seg_volume = 0
    for idx_labels, label in enumerate(Infarct_labels):
        if label == 'Background' or label == 'CSF':
            continue
        
        synseg_ml = 0
        for j_labels, label_index in enumerate(Infarct_labels[label]):
            synseg_ml += np.sum(new_SynthSeg_array == label_index) * ml_size
        
        if round(synseg_ml, 1) >= 0.3:
            new_Infarct_labels[label] = round(synseg_ml, 1)
        total_seg_volume += synseg_ml
    
    # 選擇要顯示的標籤
    Infarct_labels_show = {}
    color_num = 0
    if len(new_Infarct_labels) >= 10:
        for idx_labels, label in enumerate(new_Infarct_labels):
            if new_Infarct_labels[label] >= 1 and color_num < 15:
                Infarct_labels_show[label] = new_Infarct_labels[label]
                color_num += 1
    else:
        for idx_labels, label in enumerate(new_Infarct_labels):
            Infarct_labels_show[label] = new_Infarct_labels[label]
    
    if total_seg_volume > InfarctVolume:
        InfarctVolume = round(total_seg_volume, 1)
    
    # 生成報告文字
    if InfarctVolume > 0:
        text = f'DWI bright, ADC dark (mean ADC value: {MeanADC}) lesions, estimated about {InfarctVolume} ml'
        text2 = ''
        
        if len(new_Infarct_labels) > 0:
            text += ', involving '
            for idx_label, new_label in enumerate(new_Infarct_labels):
                text += new_label.lower() + ', '
                text2 += new_label + ': ' + str(new_Infarct_labels[new_label]) + ' ml<br>'
            
            text += 'suggestive of recent infarction.<br><br>'
            text += '===================================================================================================<br>'
        else:
            text += '.'
        
        text += text2
    else:
        text = 'No definite diffusion restricted lesion in the study.'
    
    # 決定顯示的切片
    if z_i >= 21:
        combine_num = list(range(20, 0, -1))
        combine_s_num = list(range(18, 2, -1))
    elif z_i >= 19:
        combine_num = list(range(z_i-1, -1, -1))
        combine_s_num = list(range(18, 2, -1))
    else:
        combine_num = list(range(z_i-1, -1, -1))
        combine_s_num = combine_num
    
    slice_y = int((ROWS - y_i) / 2)
    slice_x = int((COLS - x_i) / 2)
    
    # 計算背景值
    Dback_v = DWI1000_array[slice_y:slice_y+y_i, slice_x:slice_x+x_i, 0][2, 2] if y_i > 2 and x_i > 2 else 0
    
    # 生成三張 PNG 圖片（直接輸出到 path_result）
    _generate_png_images(
        y_pred_cluster, DWI1000_nor, new_SynthSeg_array, 
        Infarct_labels, Infarct_labels_show, colorbar,
        combine_num, combine_s_num, ROWS, COLS, 
        slice_y, slice_x, y_i, x_i, z_i,
        Dback_v, InfarctVolume, MeanADC,
        path_result, patient_id  # 直接輸出到 Result 資料夾
    )
    
    # 生成 JSON 報告
    path_json_out = os.path.join(path_nnunet, 'JSON')
    path_dicomseg = os.path.join(path_dcm, 'Dicom-Seg')
    path_dicom_zip = os.path.join(path_nnunet, 'Dicom_zip')
    
    for folder in [path_json_out, path_dicomseg, path_dicom_zip]:
        os.makedirs(folder, exist_ok=True)
    
    json_files = [
        os.path.join(path_json, 'Pred_Infarct.json'),
        os.path.join(path_json_out, f'{patient_id}_platform_json.json'),
        os.path.join(path_output, 'Pred_Infarct.json')
    ]
    
    for json_path in json_files:
        create_json_report(json_path, patient_id, InfarctVolume, MeanADC, text)
    
    print(f"後處理完成：梗塞體積 = {InfarctVolume} ml, 平均 ADC = {MeanADC}")
    return True


def _generate_png_images(
    y_pred_cluster, DWI1000_array, new_SynthSeg_array,
    Infarct_labels, Infarct_labels_show, colorbar,
    combine_num, combine_s_num, ROWS, COLS,
    slice_y, slice_x, y_i, x_i, z_i,
    Dback_v, InfarctVolume, MeanADC,
    path_output, patient_id
):
    """生成三張 PNG 視覺化圖片"""
    
    # 圖片 1: 無預測遮罩
    plt.style.use('dark_background')
    fig = plt.figure()
    plt.axis('off')
    
    volume_result = '\n\n(For Research Purpose Only)'
    plt.text(0.5, 0.1, volume_result, fontsize=15, 
            verticalalignment='center', horizontalalignment='center')
    
    for k in range(len(combine_num)):
        D_show = np.zeros((ROWS, COLS))
        D_show[slice_y:slice_y+y_i, slice_x:slice_x+x_i] = DWI1000_array[:, :, combine_num[k]].copy()
        min_value = np.min(D_show)
        D_show[D_show == Dback_v] = min_value
        
        predict_show = ((D_show - np.min(D_show)) * 40).copy()
        predict_show = np.stack([predict_show] * 3, axis=-1)
        predict_show = np.clip(predict_show, 0, 255).astype('uint8')
        
        row = k // 5
        col = k % 5
        ax2 = fig.add_axes([0 + 0.2*col, 0.9 - 0.2*row, 0.2, 0.2])
        ax2.imshow(predict_show)
        ax2.axis('off')
    
    plt.savefig(os.path.join(path_output, f'{patient_id}_noPred.png'), 
               bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close('all')
    
    # 圖片 2: 有預測遮罩
    fig = plt.figure()
    plt.axis('off')
    
    if np.sum(y_pred_cluster) > 0:
        volume_result = f'Infarction Volume: {InfarctVolume} ml (mean ADC = {MeanADC})\n\n(For Research Purpose Only)'
    else:
        volume_result = 'Infarction Volume: 0 ml\n\n(For Research Purpose Only)'
    
    plt.text(0.5, 0.1, volume_result, fontsize=15,
            verticalalignment='center', horizontalalignment='center')
    
    for k in range(len(combine_num)):
        y_color = y_pred_cluster[:, :, combine_num[k]].copy()
        D_show = np.zeros((ROWS, COLS))
        D_show[slice_y:slice_y+y_i, slice_x:slice_x+x_i] = DWI1000_array[:, :, combine_num[k]].copy()
        min_value = np.min(D_show)
        D_show[D_show == Dback_v] = min_value
        
        predict_show = ((D_show - np.min(D_show)) * 40).copy()
        predict_show = np.stack([predict_show] * 3, axis=-1)
        predict_show = np.clip(predict_show, 0, 255).astype('uint8')
        
        y_c, x_c = np.where(y_color > 0)
        if len(y_c) > 0:
            predict_show[y_c, x_c, 0] = 255
            predict_show[y_c, x_c, 1] = 0
            predict_show[y_c, x_c, 2] = 255
        
        row = k // 5
        col = k % 5
        ax2 = fig.add_axes([0 + 0.2*col, 0.9 - 0.2*row, 0.2, 0.2])
        ax2.imshow(predict_show)
        ax2.axis('off')
    
    plt.savefig(os.path.join(path_output, f'{patient_id}.png'),
               bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close('all')
    
    # 圖片 3: SynthSEG 分區圖
    if len(Infarct_labels_show) > 0:
        fig = plt.figure()
        plt.axis('off')
        
        if np.sum(y_pred_cluster) > 0:
            volume_result = f'Infarction Volume: {InfarctVolume} ml (mean ADC = {MeanADC})\n\n(For Research Purpose Only)'
        else:
            volume_result = 'Infarction Volume: 0 ml\n\n(For Research Purpose Only)'
        
        plt.text(0.5, 0.1, volume_result, fontsize=15,
                verticalalignment='center', horizontalalignment='center')
        
        for k in range(len(combine_s_num)):
            y_color = new_SynthSeg_array[:, :, combine_s_num[k]].copy()
            D_show = np.zeros((ROWS, COLS))
            D_show[slice_y:slice_y+y_i, slice_x:slice_x+x_i] = DWI1000_array[:, :, combine_s_num[k]].copy()
            min_value = np.min(D_show)
            D_show[D_show == Dback_v] = min_value
            
            predict_show = ((D_show - np.min(D_show)) * 40).copy()
            predict_show = np.stack([predict_show] * 3, axis=-1)
            predict_show = np.clip(predict_show, 0, 255).astype('uint8')
            
            for idx_labels, label_one in enumerate(Infarct_labels_show):
                for jdx_labels, label_num in enumerate(Infarct_labels[label_one]):
                    y_c, x_c = np.where(y_color == label_num)
                    if len(y_c) > 0:
                        rgb = mcolors.to_rgb(colorbar[label_one])
                        predict_show[y_c, x_c, 0] = rgb[0] * 255
                        predict_show[y_c, x_c, 1] = rgb[1] * 255
                        predict_show[y_c, x_c, 2] = rgb[2] * 255
            
            row = k // 4
            col = k % 4
            ax2 = fig.add_axes([0 + 0.16*col, 0.9 - 0.2*row, 0.21, 0.21])
            ax2.imshow(predict_show)
            ax2.axis('off')
        
        # 添加色標
        ax2 = fig.add_axes([0 + 0.165*4, 0.315+0.4*(1-len(Infarct_labels_show)/15), 
                           0.015, 0.05*len(Infarct_labels_show)])
        for idx_labels, label in enumerate(Infarct_labels_show):
            ax2.hlines(1-0.03*idx_labels, 0, 1, color=colorbar[label], linewidth=6)
            ax2.text(1.3, 0.996-0.03*idx_labels, 
                    f'{label}: {Infarct_labels_show[label]} ml',
                    fontsize=6, horizontalalignment='left')
            ax2.axis('off')
        
        plt.savefig(os.path.join(path_output, f'{patient_id}_SynthSEG.png'),
                   bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close('all')

