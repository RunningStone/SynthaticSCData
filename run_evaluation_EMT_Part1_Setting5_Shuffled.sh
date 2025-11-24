#!/bin/bash
# 为EMT_Part1_Setting5_Shuffled生成Evaluation结果
# 
# 实验5：时间信息解耦实验（时间打乱）
# 目的：测试模型是否真正学习了时间依赖动力学，还是仅仅记忆了高维空间中的插值映射
# 
# 使用方法：
#   bash run_evaluation_EMT_Part1_Setting5_Shuffled.sh

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "EMT Part1 Setting5 Shuffled - Re-evaluation"
echo "========================================================================"
echo ""
echo "实验类型: 时间信息解耦实验（打乱时序组）"
echo "实验配置: experiment_EMT_Part1_setting5_shuffled.yaml"
echo "输出目录: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting5_Shuffled"
echo ""
echo "理论背景:"
echo "  - 破坏时间因果顺序，仅保留状态空间几何信息"
echo "  - 检验Schrödinger Bridge是否真正学习了时间依赖漂移场 b(x,t)"
echo "  - 性能差异量化因果信息贡献: ΔP ∝ I_causal(T)"
echo ""

# 配置参数
CONFIG_FILE="experiment_EMT_Part1_setting5_shuffled.yaml"
CHECKPOINT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting5_Shuffled/checkpoints"
OUTPUT_SUFFIX="_fixed"

# 检查checkpoint目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ 错误：Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

# 检查模型文件
echo "检查模型checkpoint..."
MODELS_FOUND=0
for model in sb_mlplus; do
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
    echo "  - JSON: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting5_Shuffled/results${OUTPUT_SUFFIX}.json"
    echo "  - 日志: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting5_Shuffled/logs/evaluation_only${OUTPUT_SUFFIX}.log"
    echo ""
    echo "🔬 下一步分析建议："
    echo "  1. 对比Setting3（正常时序）vs Setting5（打乱时序）的性能差异"
    echo "  2. 量化因果信息贡献: ΔP = P_ordered - P_shuffled"
    echo "  3. 如果性能崩溃 → 验证SB学习了时间依赖动力学"
    echo "  4. 如果性能保持 → SB可能退化为条件生成器"
    echo ""
    echo "你可以使用以下命令查看结果："
    echo "  cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting5_Shuffled/results${OUTPUT_SUFFIX}.json | jq ."
else
    echo "❌ Evaluation失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "请检查日志文件了解详细错误信息："
    echo "  /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting5_Shuffled/logs/evaluation_only${OUTPUT_SUFFIX}.log"
fi
echo "========================================================================"

exit $EXIT_CODE
