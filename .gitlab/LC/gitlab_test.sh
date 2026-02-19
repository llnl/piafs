#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

echo "Start: $(date)"

echo "========="
echo "GitLab CI"
echo "========="

modules=${MODULE_LIST:-""}
mpiexec_executable=${MPIEXEC_EXECUTABLE:-"srun"}
# If using flux, append "run" after the flux executable path
if [[ "${mpiexec_executable}" == "flux" ]]
then
    mpiexec_executable="$(which ${mpiexec_executable}) run"
    flux jobs
    flux resource list
else
    mpiexec_executable="$(which ${mpiexec_executable})"
fi

mpiexec_preflags=${MPIEXEC_PREFLAGS:-""}
host=$(hostname)
build_type=${BUILD_TYPE:-"Release"}

PIAFS_ENABLE_CUDA=${PIAFS_ENABLE_CUDA:-"OFF"}
PIAFS_ENABLE_HIP=${PIAFS_ENABLE_HIP:-"OFF"}

basehost=${host//[[:digit:]]/}

echo "HOST: ${host}"
src_dir="${PWD}"
echo "Source directory: ${src_dir}"
build_dir="$(realpath -- "${src_dir}/../build_${host}_${CI_PIPELINE_ID}_$(date +%F_%H_%M_%S)")"
echo "Build directory: ${build_dir}"

echo "============="
echo "Setup modules"
echo "============="

if [[ -n ${modules} ]]
then
    module load ${modules}
fi
module list

echo "================="
echo "Configure PIAFS"
echo "================="

# Base CMake options
CMAKE_OPTS=(
    -G Ninja
    -S "${src_dir}"
    -B "${build_dir}"
    -D CMAKE_INSTALL_PREFIX:PATH="${build_dir}/install"
    -D CMAKE_CXX_COMPILER:STRING="${CMAKE_CXX_COMPILER:-mpicxx}"
    -D CMAKE_C_COMPILER:STRING="${CMAKE_C_COMPILER:-mpicc}"
    -D CMAKE_BUILD_TYPE:STRING="${build_type}"
    -D MPIEXEC:STRING="${mpiexec_executable}"
    -D CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
)

# GPU options
if [[ "${PIAFS_ENABLE_CUDA}" == "ON" ]]
then
    CMAKE_OPTS+=(
        -D ENABLE_GPU:BOOL=ON
        -D ENABLE_CUDA:BOOL=ON
        -D CMAKE_CUDA_ARCHITECTURES:STRING="${CUDA_ARCH:-75}"
    )
    echo "CUDA enabled with architecture: ${CUDA_ARCH:-75}"
elif [[ "${PIAFS_ENABLE_HIP}" == "ON" ]]
then
    CMAKE_OPTS+=(
        -D ENABLE_GPU:BOOL=ON
        -D ENABLE_HIP:BOOL=ON
        -D CMAKE_HIP_ARCHITECTURES:STRING="${AMD_ARCH:-gfx942}"
    )
    echo "HIP enabled with architecture: ${AMD_ARCH:-gfx942}"
fi

time cmake "${CMAKE_OPTS[@]}"

echo "============="
echo "Build PIAFS"
echo "============="

time cmake --build "${build_dir}" --parallel

echo "============"
echo "Test PIAFS"
echo "============"

# Run tests (unit tests and regression tests)
time ctest --test-dir "${build_dir}" --output-on-failure

echo "End: $(date)"
