FROM tf_2_14:latest

# Replace CPU onnxruntime with GPU version (1.17.1 is the last version that works
# with CUDA 12.4 + TF 2.14 without numpy conflicts).
# --no-deps prevents pip from touching numpy/TF which would break TF 2.14.
COPY onnxruntime_gpu-1.17.1-cp310-cp310-manylinux_2_28_x86_64.whl /tmp/
RUN /opt/conda/envs/tf_2_14/bin/python -m pip uninstall -y onnxruntime && \
    /opt/conda/envs/tf_2_14/bin/python -m pip install --no-deps /tmp/onnxruntime_gpu-1.17.1-cp310-cp310-manylinux_2_28_x86_64.whl && \
    rm /tmp/onnxruntime_gpu-1.17.1-cp310-cp310-manylinux_2_28_x86_64.whl
