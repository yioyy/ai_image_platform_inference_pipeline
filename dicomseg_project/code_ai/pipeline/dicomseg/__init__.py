import os
import pathlib
import pydicom
import subprocess

from code_ai import PYTHON3

EXAMPLE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'resource', 'SEG_20230210_160056_635_S3.dcm')
DCM_EXAMPLE = pydicom.dcmread(EXAMPLE_FILE)


def dicom_seg_multi_file(ID:str,
                         InputsDicomDir:str,
                         nii_path_str:str,
                         path_output:str):
    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/dicomseg/create_dicomseg_multi_file_json_claude.py '
               '--ID {} '
               '--InputsDicomDir {} '
               '--Inputs {} '
               '--Output_folder {} '.format(pathlib.Path(__file__).parent.parent.parent.parent.absolute(),
                                            PYTHON3,
                                            ID,
                                            InputsDicomDir,
                                            nii_path_str,
                                            path_output)
               )

    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print('dicom_seg_multi_file',cmd_str)
    return stdout, stderr


def dicom_seg_cmb_file(ID:str,
                         InputsDicomDir:str,
                         nii_path_str:str,
                         path_output:str):
    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/dicomseg/cmb.py '
               '--ID {} '
               '--InputsDicomDir {} '
               '--Inputs {} '
               '--Output_folder {} '.format(pathlib.Path(__file__).parent.parent.parent.parent.absolute(),
                                            PYTHON3,
                                            ID,
                                            InputsDicomDir,
                                            nii_path_str,
                                            path_output)
               )

    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print('dicom_seg_cmb_file',cmd_str)
    return stdout, stderr