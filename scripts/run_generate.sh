#!/bin/bash
# 建立虛擬環境並執行圖像生成腳本
# 用法: ./scripts/run_generate.sh [-d|-r|-t [test_type]] [表情]
# 選項:
#   -d: 使用預設動作和表情 (默認)
#   -r: 使用隨機動作和表情
#   -t: 使用測試提示詞模式，可選參數 test_type:
#       basic: 基本外觀測試 (默認)
#       side: 側面視圖測試
#       back: 背面視圖測試
#       action: 動作測試
#       expression: 表情測試
# 參數:
#   表情: 可選，指定表情，默認為 "smiling"

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
ENV_DIR=".venv_mushroom"
PY_SCRIPT="src/main.py"
CONFIG_FILE="src/config/config.yaml"
REQUIRED_PACKAGES="diffusers transformers accelerate safetensors peft rembg pyyaml onnxruntime"

# 日誌函數
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 檢查依賴
check_dependency() {
    command -v $1 >/dev/null 2>&1 || { log_error "$1 未安裝，請先安裝它。"; exit 1; }
}

# 檢查 Python 和 pip
check_dependency python3
check_dependency pip

# 在執行腳本前設置 PYTHONPATH
export PYTHONPATH=$(pwd)

# 建立虛擬環境（如不存在）
if [ ! -d "$ENV_DIR" ]; then
    log_info "創建虛擬環境 $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    log_info "升級 pip..."
    pip install --upgrade pip
    log_info "安裝依賴包..."
    pip install $REQUIRED_PACKAGES
    log_success "虛擬環境設置完成"
else
    log_info "使用現有虛擬環境 $ENV_DIR"
    source "$ENV_DIR/bin/activate"
fi

# 檢查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "找不到配置文件: $CONFIG_FILE"
    exit 1
fi

# 處理命令行參數
EXPRESSION=${1:-"smiling"}  # 使用第一個參數作為表情，默認為 smiling
MODE="-d"  # 默認使用預設動作和表情
TEST_TYPE="basic"  # 默認測試類型

while getopts ":drt:" opt; do
    case $opt in
        d)
            MODE="-d"
            log_info "使用預設動作和表情"
            ;;
        r)
            MODE="-r"
            log_info "使用隨機動作和表情"
            ;;
        t)
            MODE="-t"
            TEST_TYPE=${OPTARG:-"basic"}
            log_info "使用測試提示詞模式，測試類型: $TEST_TYPE"
            ;;
        *)
            log_warning "未知選項: -$OPTARG，使用預設動作和表情"
            MODE="-d"
            ;;
    esac
done

# 開始生成圖像
log_info "開始生成圖像..."
log_info "使用表情: $EXPRESSION"
log_info "使用模式: $MODE"

# 執行 Python 腳本
if [ "$MODE" == "-t" ]; then
    log_info "執行測試模式，測試類型: $TEST_TYPE"
    python3 "$PY_SCRIPT" $MODE $TEST_TYPE
else
    python3 "$PY_SCRIPT" $MODE $EXPRESSION
fi

# 檢查腳本執行結果
if [ $? -eq 0 ]; then
    log_success "所有圖像生成完成！"
else
    log_error "圖像生成過程中發生錯誤"
    exit 1
fi