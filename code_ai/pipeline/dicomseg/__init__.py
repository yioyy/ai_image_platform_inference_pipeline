import os
import pydicom


EXAMPLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'resource', 'SEG_20230210_160056_635_S3.dcm')
DCM_EXAMPLE = pydicom.dcmread(EXAMPLE_FILE)
