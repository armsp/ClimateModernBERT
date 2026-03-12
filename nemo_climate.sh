#!/usr/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --mem=512G
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus=H100:1
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err

source ~/hpc_notify.sh
source .venv/bin/activate
module load dev2025a h100 cuda/12.9.1

# export LD_LIBRARY_PATH=/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/gcc-13.2.0-hgptpx2eoraipaxlrxijwyj5jxznibqq/lib64:$LD_LIBRARY_PATH
export HF_HOME="/scratch/sraj/"
export TRITON_CACHE_DIR="/scratch/sraj/tritoncache"
# or
export TRITON_HOME="/scratch/sraj/tritoncache"
export VLLM_CACHE_ROOT="/scratch/sraj/vllmcache"
TOKEN_FILE="./hf_token.txt"

if [[ ! -f "${TOKEN_FILE}" ]]; then
	echo "Token file not found at ${TOKEN_FILE}." >&2
	exit 1
fi

read -r HF_TOKEN < "${TOKEN_FILE}"
HF_TOKEN="${HF_TOKEN//$'\r'}"
HF_TOKEN="${HF_TOKEN//$'\n'}"
if [[ -z "${HF_TOKEN}" ]]; then
	echo "Token file ${TOKEN_FILE} is empty." >&2
	exit 1
fi
export HF_TOKEN
# uv pip install torch wheel_stub psutil setuptools setuptools_scm
# echo "transformers==4.55.2" > override.txt
# uv pip install  https://pypi.nvidia.com --no-build-isolation "nemo-curator[all]" --override override.txt
# uv pip install nemo-curator[deduplication_cuda12]

# cd /home/sraj/data/mbert/nemo_curator
# python generate_test_data.py                                    # already done — data in input_data/
hpc_notify "🚀 Started: Deduplication"

# python nemo_pipeline_climate.py --input-dir synthetic_data_1/ --output-dir synthetic_data_1_nemo/        # runs all 3 stages

python nemo_pipeline_climate.py --input-dir markus_1/ --output-dir markus_1_nemo/        # runs all 3 stages


hpc_notify "✅ Finished: Synthetic Data Deduplication"