# 脉动阵列加法器网络模拟器

这是一个用Python实现的脉动阵列加法器网络模拟器，可用于DNN加速器的性能评估。模拟器提供了能耗(pJ)、面积(mm²)和延迟(cycle)的性能评估功能。

## 功能特点

- 可配置的脉动阵列大小（行数和列数）
- 可配置的数据位宽
- 支持向量加法和矩阵加法
- 性能评估指标：
  - 能耗 (pJ)
  - 面积 (mm²)
  - 延迟 (cycles和ns)
  - 能效 (TOPS/W)
- 可视化脉动阵列状态

## 使用方法

### 基本用法

```bash
# 运行矩阵加法示例（默认）
python systolic_adder_simulator.py

# 运行向量加法示例
python systolic_adder_simulator.py --example vector

# 运行自定义大小的矩阵加法
python systolic_adder_simulator.py --example custom --rows 8 --cols 8 --bit_width 32 --visualize
```

### 命令行参数

- `--rows`: 脉动阵列行数（默认：4）
- `--cols`: 脉动阵列列数（默认：4）
- `--bit_width`: 数据位宽（默认：16）
- `--example`: 示例类型，可选 'vector'、'matrix' 或 'custom'（默认：'matrix'）
- `--visualize`: 是否生成可视化图像（默认：不生成）

### 示例代码

```python
from systolic_adder_simulator import SystolicArray
import numpy as np

# 创建一个4x4的脉动阵列
systolic = SystolicArray(rows=4, cols=4, bit_width=16)

# 准备输入数据
matrix_a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

matrix_b = np.array([
    [16, 15, 14, 13],
    [12, 11, 10, 9],
    [8, 7, 6, 5],
    [4, 3, 2, 1]
])

# 设置输入格式（对角线输入方式）
horizontal_inputs = [[0 for _ in range(8)] for _ in range(4)]
vertical_inputs = [[0 for _ in range(4)] for _ in range(8)]

# 将矩阵A沿对角线输入
for i in range(4):
    for j in range(4):
        horizontal_inputs[i][i+j] = matrix_a[i][j]

# 将矩阵B沿对角线输入
for i in range(4):
    for j in range(4):
        vertical_inputs[i+j][j] = matrix_b[i][j]

# 设置输入
systolic.set_inputs(horizontal_inputs, vertical_inputs)

# 运行模拟
systolic.run_simulation(num_cycles=8)

# 获取结果
results = systolic.outputs

# 打印性能指标
metrics = systolic.get_performance_metrics()
print(f"能耗: {metrics['energy_pJ']:.2f} pJ")
print(f"面积: {metrics['area_mm2']:.6f} mm²")
print(f"延迟: {metrics['latency_cycles']} cycles")
print(f"能效: {metrics['energy_efficiency_TOPS_per_W']:.6f} TOPS/W")
```

## 模拟器架构

模拟器由以下主要组件组成：

1. **处理单元 (PE)**: 实现基本的加法操作，并跟踪能耗和面积
2. **脉动阵列**: 由多个PE组成的二维网格，实现数据流动和计算
3. **性能评估模块**: 计算整个阵列的能耗、面积和延迟

## 性能模型

- **能耗模型**: 基于操作数量和位宽，使用典型的16nm工艺参数
- **面积模型**: 基于PE数量和位宽，面积与位宽的平方成正比
- **延迟模型**: 基于周期数和时钟周期

## 可视化

模拟器可以生成每个周期的脉动阵列状态可视化图像，显示数据在阵列中的流动和计算过程。

## 注意事项

- 这是一个功能性模拟器，主要用于教学和研究目的
- 性能参数基于典型的16nm工艺，可以根据实际需求进行调整
- 模拟器假设理想的数据流动，没有考虑实际硬件中的各种延迟和开销

