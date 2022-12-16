SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PROJECT_PATH=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}
# export ADACRS_DIR=
python ${PROJECT_PATH}/adacrs/utils/preprocess.py --data_name INS