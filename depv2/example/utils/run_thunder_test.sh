#!/bin/bash

echo "==============================================="
echo "Thunder Robot Interface Test Runner"
echo "==============================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

# 检查必要的Python包
echo "🔍 Checking Python dependencies..."
python3 -c "import numpy, asyncio, threading" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required Python packages (numpy, asyncio, threading)"
    echo "   Please install: pip3 install numpy"
    exit 1
fi

echo "✅ Python environment OK"

# 进入正确的目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# 检查测试文件是否存在
if [ ! -f "test_thunder_interface.py" ]; then
    echo "❌ test_thunder_interface.py not found in current directory"
    exit 1
fi

# 检查上级目录中的thunder_robot_interface.py
if [ ! -f "../thunder_robot_interface.py" ]; then
    echo "❌ thunder_robot_interface.py not found in parent directory"
    echo "   Expected: ../thunder_robot_interface.py"
    exit 1
fi

echo "✅ All test files found"

# 运行测试
echo ""
echo "🚀 Starting Thunder Robot Interface Tests..."
echo "==============================================="

# 使用unbuffered output以便实时看到输出
export PYTHONUNBUFFERED=1

# 运行测试
python3 test_thunder_interface.py

# 获取退出状态
exit_code=$?

echo ""
echo "==============================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Test completed successfully!"
else
    echo "❌ Test completed with errors (exit code: $exit_code)"
fi
echo "==============================================="

exit $exit_code 