SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PROJECT_PATH=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH=$PYTHONPATH:${PROJECT_PATH}
python ${PROJECT_PATH}/adacrs/train.py --data_name LAST_FM
