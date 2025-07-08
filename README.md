# Thunder Robot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-JIT-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

##  系统架构

```
Thunder Robot Deployment System v2.1.0
├── depv2/                          # 主项目目录
│   ├── run_deploy.py              # 🚀 主部署脚本
│   ├── run_monitor.py             # 📊 硬件监控工具
│   ├── lib/                       # 📚 核心库组件
│   │   ├── core/                  # 核心管理模块
│   │   │   ├── thunder_deployer.py       # 主协调器
│   │   │   ├── deployment_config.py      # 配置管理
│   │   │   ├── model_manager.py          # 模型加载与推理
│   │   │   ├── deployment_state.py       # 状态管理
│   │   │   ├── statistics_collector.py   # 性能统计
│   │   │   └── graceful_stop_manager.py  # 优雅停止控制
│   │   ├── hardware/              # 硬件接口层
│   │   ├── safety/                # 安全监控层
│   │   └── config/                # 配置管理层
│   ├── scripts/                   # 🛠️ 实用工具
│   ├── example/                   # 📘 示例和测试
│   ├── config/                    # ⚙️ 配置文件
│   └── tools/                     # 🔧 开发工具
```


## 快速开始

### 1. 环境要求

```bash
# Python环境
Python 3.8+

# 依赖包
pip install torch numpy scipy pyyaml pyserial
```

### 2. 硬件连接

- **机器人**: Thunder四足轮式机器人，IP: `192.168.66.159:12345`
- **IMU传感器**: Hipnuc IMU连接至 `/dev/ttyUSB0`，波特率 115200
- **控制频率**: 50Hz策略推理，200Hz状态接收

### 3. 部署模型

将训练好的PyTorch JIT模型放置在 `depv2/exported/policy.pt`

### 4. 运行系统

####  自动模式（默认安全模式）
```bash
cd deploy_dog/depv2
python run_deploy.py
```
*默认以安全模式运行，不会发送实际的电机指令，适合测试*

#### 交互模式（可选择安全/主动模式）
```bash
cd deploy_dog/depv2
python run_deploy.py --interactive
```

**快捷控制命令：**
```
系统控制:                    运动控制:
  S - 启动安全模式            0 - 停止 (0,0,0)
  A - 启动主动模式            w - 前进 (0.5,0,0)
  x - 优雅停止               s - 后退 (-0.5,0,0)
  ! - 立即停止               a - 左移 (0,-0.5,0)
  ? - 显示状态               d - 右移 (0,0.5,0)
  i - 显示统计               e - 右转 (0,0,-0.5)
  q - 退出                   r - 左转 (0,0,0.5)
```

#### 硬件监控
```bash
cd deploy_dog/depv2
python run_monitor.py
```

## 🛠️ 主要功能

### 1. 智能部署

```python
from lib.core.thunder_deployer import ThunderDeployer

# 初始化部署器
deployer = ThunderDeployer("exported/policy.pt", "config/thunder_flat_config.yaml")

# 设置速度指令并启动
velocity_commands = deployer.set_velocity_commands(1.0, 0.0, 0.0)  # 前进
await deployer.start(velocity_commands, safe_mode=True)
```

### 2. 硬件诊断

```bash
# 全面硬件检查
python scripts/check_hardware.py

# IMU校准监控
python scripts/monitor_imu_calibration.py

# 调试IMU数据
python scripts/debug_imu_data.py
```

### 3. 性能监控

```python
# 获取系统状态
current_state = deployer.get_current_state()
print(f"性能评分: {current_state['performance_score']:.1f}%")

# 获取性能统计
performance = deployer.get_performance_summary()
print(f"平均频率: {performance['basic_metrics']['avg_frequency']:.1f}Hz")
```

## 机器人规格

### 硬件配置
- **结构**: 四足轮式混合机器人
- **关节**: 16个自由度（12个腿部关节 + 4个轮子）
- **控制**: 位置控制（腿部）+ 速度控制（轮子）

### 模型
- **输入**: 57维观测向量
  - 基座角速度 (3维, IMU数据)
  - 重力投影向量 (3维, IMU计算)
  - 速度指令 (3维)
  - 关节位置 (16维)
  - 关节速度 (16维)
  - 上一步动作 (16维)
- **输出**: 16维动作向量
- **频率**: 50Hz推理
- **格式**: PyTorch JIT (.pt)

### 控制特性
```yaml
关节类型     | 数量 | 缩放因子 | 控制模式
-----------|------|---------|----------
Hip关节     | 4    | ×0.125  | 位置控制
Thigh关节   | 4    | ×0.25   | 位置控制  
Calf关节    | 4    | ×0.25   | 位置控制
Wheel关节   | 4    | ×5.0    | 速度控制
```

## ⚙️ 配置说明

主配置文件: `depv2/config/thunder_flat_config.yaml`

### 核心配置项

```yaml
robot:
  control_frequency: 50.0    # 控制频率
  policy_frequency: 50.0     # 策略频率

observations:
  observation_dim: 57        # 观测维度
  scales:
    base_ang_vel: 0.25      # IMU角速度缩放
    joint_vel: 0.05         # 关节速度缩放

actions:
  action_dim: 16            # 动作维度
  scales:
    hip_joints: 0.125       # Hip关节缩放
    wheel_joints: 5.0       # 轮子关节缩放

safety:
  max_tilt_angle: 45.0      # 最大倾斜角度
  emergency_stop_enabled: true

如遇问题，请检查：

1. **硬件连接**: IMU (`/dev/ttyUSB0`) 和机器人 (`192.168.66.159:12345`)
2. **模型文件**: `depv2/exported/policy.pt` 是否存在
3. **配置文件**: `depv2/config/thunder_flat_config.yaml` 是否正确
4. **系统日志**: 查看 `depv2/logs/` 目录下的日志文件
5. **硬件诊断**: 运行 `python scripts/check_hardware.py`

