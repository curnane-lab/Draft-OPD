cd ./sglang-dflash
pip install -e "./python"
pip install cachetools
cd ./../verl
pip install -e .

# --- Ascend NPU (optional) -------------------------------------------------
# Run with WITH_NPU=1 to install the Ascend NPU runtime dependencies for the
# sglang-dflash engine. Adjust the versions to match your CANN toolkit; see
# sglang-dflash/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx
# for the supported matrix.
#
#   WITH_NPU=1 PYTORCH_VERSION=2.7.1 TORCHVISION_VERSION=0.22.1 \
#     TORCH_NPU_VERSION=2.7.1 bash install.sh
if [ "${WITH_NPU:-0}" = "1" ]; then
    PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.1}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
    TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.7.1}"
    pip install torch=="$PYTORCH_VERSION" torchvision=="$TORCHVISION_VERSION" \
        --index-url https://download.pytorch.org/whl/cpu
    pip install torch_npu=="$TORCH_NPU_VERSION"
    # sgl-kernel-npu provides fused kernels used by the DFlash NPU path
    # (e.g. split_qkv_rmsnorm_rope). See https://github.com/sgl-project/sgl-kernel-npu
    pip install sgl-kernel-npu || echo "WARNING: sgl-kernel-npu install failed; DFlash will use eager fallbacks on NPU."
fi
