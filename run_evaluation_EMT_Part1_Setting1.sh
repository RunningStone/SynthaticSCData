#!/bin/bash
# 为EMT_Part1_Setting1生成Evaluation结果
# 
# 用途：修复OOM问题后重新生成evaluation metrics
# 
# 使用方法：
#   bash run_evaluation_EMT_Part1_Setting1.sh

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "EMT Part1 Setting1 - Re-evaluation"
echo "========================================================================"
echo ""
echo "实验配置: experiment_EMT_Part1_setting1.yaml"
echo "输出目录: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1"
echo ""

# 配置参数
CONFIG_FILE="experiment_EMT_Part1_setting1.yaml"
CHECKPOINT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1/checkpoints"
OUTPUT_SUFFIX="_fixed"

# 检查checkpoint目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ 错误：Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

# 检查模型文件
echo "检查模型checkpoint..."
MODELS_FOUND=0
for model in sb ot vae; do
    if [ -f "$CHECKPOINT_DIR/$model/final_model.pt" ]; then
        echo "  ✓ 找到 $model/final_model.pt"
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        echo "  ✗ 未找到 $model/final_model.pt"
    fi
done

if [ $MODELS_FOUND -eq 0 ]; then
    echo "❌ 错误：未找到任何模型checkpoint"
    exit 1
fi

echo ""
echo "找到 $MODELS_FOUND 个模型，开始评估..."
echo "========================================================================"
echo ""

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "激活虚拟环境 (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "激活虚拟环境 (venv)..."
    source venv/bin/activate
fi

# 运行evaluation
python step1_run_evaluation_only.py \
    "$CONFIG_FILE" \
    "$CHECKPOINT_DIR" \
    --config_dir configs \
    --output_suffix "$OUTPUT_SUFFIX"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation完成！"
    echo ""
    echo "结果文件："
    echo "  - JSON: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1/results${OUTPUT_SUFFIX}.json"
    echo "  - 日志: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1/logs/evaluation_only${OUTPUT_SUFFIX}.log"
    echo ""
    echo "你可以使用以下命令查看结果："
    echo "  cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1/results${OUTPUT_SUFFIX}.json | jq ."
else
    echo "❌ Evaluation失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "请检查日志文件了解详细错误信息："
    echo "  /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1/logs/evaluation_only${OUTPUT_SUFFIX}.log"
fi
echo "========================================================================"

exit $EXIT_CODE
