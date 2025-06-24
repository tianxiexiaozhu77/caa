
export CUDA_VISIBLE_DEVICES=2

# Python 解释器路径
PYTHON_PATH="/opt/conda/envs/caa/bin/python"

# 主程序路径
SCRIPT_PATH="/opt/data/private/zjx/ICL/CAA-main/prompting_with_steering.py"

# 日志文件
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prompting_mawps$(date +'%Y%m%d_%H%M%S').log"

# 参数
ARGS=(
  "--layers" 0 1 2 3 4 5 6 7 \
           8 9 10 11 12 13 14 15 \
           16 17 18 19 20 21 22 23 \
           24 25 26 27 28 29 30 31
  "--multipliers" -1 0 1
  "--type" "open_ended"
  "--model_size" "7b"
  "--behavior" "mawps"
  "--model" "/opt/data/private/zjx/ICL/inform_zjx/Llama-2-7b-chat-hf"
)

# 后台运行并写入日志
nohup "$PYTHON_PATH" "$SCRIPT_PATH" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &

# 显示运行信息
echo "Script is running in the background. Log: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' to monitor output."

