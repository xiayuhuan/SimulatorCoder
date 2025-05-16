#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脉动阵列加法器网络模拟器
包含能耗(pJ)、面积(mm²)和延迟(cycle)的性能评估

作者: Codegen
日期: 2025-05-16
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Any
import argparse

class ProcessingElement:
    """处理单元类，代表脉动阵列中的一个PE"""
    
    def __init__(self, pe_id: Tuple[int, int], bit_width: int = 16):
        """
        初始化处理单元
        
        参数:
            pe_id: PE的坐标 (row, col)
            bit_width: 数据位宽，默认16位
        """
        self.pe_id = pe_id
        self.bit_width = bit_width
        self.accumulator = 0
        self.input_a = 0
        self.input_b = 0
        self.output = 0
        self.busy = False
        self.cycles_active = 0
        self.num_operations = 0
        
    def compute(self) -> int:
        """执行加法操作并返回结果"""
        if self.busy:
            self.accumulator = self.input_a + self.input_b
            self.output = self.accumulator
            self.cycles_active += 1
            self.num_operations += 1
            return self.output
        return 0
    
    def set_inputs(self, a: int, b: int) -> None:
        """设置输入值"""
        self.input_a = a
        self.input_b = b
        self.busy = True
    
    def reset(self) -> None:
        """重置处理单元状态"""
        self.accumulator = 0
        self.input_a = 0
        self.input_b = 0
        self.output = 0
        self.busy = False
    
    def get_energy_consumption(self, energy_per_op: float) -> float:
        """
        计算能耗 (pJ)
        
        参数:
            energy_per_op: 每次操作的能耗 (pJ)
        """
        return self.num_operations * energy_per_op * (self.bit_width / 16)
    
    def get_area(self, base_area: float) -> float:
        """
        计算面积 (mm²)
        
        参数:
            base_area: 基础面积 (mm²)
        """
        # 面积与位宽的平方成正比
        return base_area * (self.bit_width / 16)**2


class SystolicArray:
    """脉动阵列类，由多个PE组成"""
    
    def __init__(self, rows: int, cols: int, bit_width: int = 16):
        """
        初始化脉动阵列
        
        参数:
            rows: 行数
            cols: 列数
            bit_width: 数据位宽，默认16位
        """
        self.rows = rows
        self.cols = cols
        self.bit_width = bit_width
        self.array = [[ProcessingElement((i, j), bit_width) for j in range(cols)] for i in range(rows)]
        self.cycle_count = 0
        self.horizontal_inputs = [[0 for _ in range(cols)] for _ in range(rows)]
        self.vertical_inputs = [[0 for _ in range(cols)] for _ in range(rows)]
        self.outputs = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # 性能参数 (基于16nm工艺的典型值)
        self.energy_per_add_op = 0.1  # pJ per operation
        self.base_pe_area = 0.001     # mm² per PE (16-bit)
        self.clock_period = 1.0       # ns
    
    def reset(self) -> None:
        """重置整个脉动阵列"""
        for i in range(self.rows):
            for j in range(self.cols):
                self.array[i][j].reset()
        self.cycle_count = 0
        self.horizontal_inputs = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.vertical_inputs = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.outputs = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
    
    def set_inputs(self, horizontal_inputs: List[List[int]], vertical_inputs: List[List[int]]) -> None:
        """
        设置输入数据
        
        参数:
            horizontal_inputs: 水平输入数据
            vertical_inputs: 垂直输入数据
        """
        self.horizontal_inputs = horizontal_inputs
        self.vertical_inputs = vertical_inputs
    
    def propagate_data(self) -> None:
        """在脉动阵列中传播数据"""
        # 保存当前输出，用于下一个周期的输入
        prev_horizontal = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        prev_vertical = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # 第一个周期的输入
        if self.cycle_count == 0:
            for i in range(self.rows):
                for j in range(self.cols):
                    if i == 0:
                        prev_horizontal[i][j] = self.horizontal_inputs[i][j]
                    if j == 0:
                        prev_vertical[i][j] = self.vertical_inputs[i][j]
        
        # 更新每个PE的输入
        for i in range(self.rows):
            for j in range(self.cols):
                # 水平输入 (来自左侧或输入)
                if j == 0:
                    # 第一列从输入获取
                    if self.cycle_count < len(self.horizontal_inputs):
                        a_input = self.horizontal_inputs[i][self.cycle_count] if self.cycle_count < len(self.horizontal_inputs[i]) else 0
                    else:
                        a_input = 0
                else:
                    # 其他列从左侧PE获取
                    a_input = self.array[i][j-1].output
                
                # 垂直输入 (来自上方或输入)
                if i == 0:
                    # 第一行从输入获取
                    if self.cycle_count < len(self.vertical_inputs):
                        b_input = self.vertical_inputs[self.cycle_count][j] if self.cycle_count < len(self.vertical_inputs) and j < len(self.vertical_inputs[self.cycle_count]) else 0
                    else:
                        b_input = 0
                else:
                    # 其他行从上方PE获取
                    b_input = self.array[i-1][j].output
                
                # 设置PE的输入
                self.array[i][j].set_inputs(a_input, b_input)
        
        # 执行计算
        for i in range(self.rows):
            for j in range(self.cols):
                self.outputs[i][j] = self.array[i][j].compute()
        
        self.cycle_count += 1
    
    def run_simulation(self, num_cycles: int) -> List[List[int]]:
        """
        运行模拟指定的周期数
        
        参数:
            num_cycles: 要模拟的周期数
        
        返回:
            最终输出结果
        """
        for _ in range(num_cycles):
            self.propagate_data()
        return self.outputs
    
    def get_total_energy(self) -> float:
        """计算总能耗 (pJ)"""
        total_energy = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                total_energy += self.array[i][j].get_energy_consumption(self.energy_per_add_op)
        return total_energy
    
    def get_total_area(self) -> float:
        """计算总面积 (mm²)"""
        total_area = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                total_area += self.array[i][j].get_area(self.base_pe_area)
        return total_area
    
    def get_latency(self) -> int:
        """获取延迟 (cycles)"""
        return self.cycle_count
    
    def get_latency_ns(self) -> float:
        """获取延迟 (ns)"""
        return self.cycle_count * self.clock_period
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "energy_pJ": self.get_total_energy(),
            "area_mm2": self.get_total_area(),
            "latency_cycles": self.get_latency(),
            "latency_ns": self.get_latency_ns(),
            "operations": sum(pe.num_operations for row in self.array for pe in row),
            "energy_efficiency_TOPS_per_W": self.calculate_energy_efficiency()
        }
    
    def calculate_energy_efficiency(self) -> float:
        """计算能效 (TOPS/W)"""
        total_ops = sum(pe.num_operations for row in self.array for pe in row)
        total_energy_joules = self.get_total_energy() * 1e-12  # 转换为焦耳
        
        if total_energy_joules == 0:
            return 0
        
        # 计算每秒可执行的操作数 (假设时钟频率为1GHz)
        clock_freq_hz = 1e9
        ops_per_second = total_ops / (self.cycle_count / clock_freq_hz)
        
        # 计算功耗 (W)
        power_watts = total_energy_joules / (self.cycle_count / clock_freq_hz)
        
        if power_watts == 0:
            return 0
        
        # 计算能效 (TOPS/W)
        energy_efficiency = (ops_per_second / 1e12) / power_watts
        
        return energy_efficiency
    
    def visualize_array(self, cycle: int = None) -> None:
        """
        可视化脉动阵列状态
        
        参数:
            cycle: 要可视化的周期，默认为当前周期
        """
        if cycle is None:
            cycle = self.cycle_count
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Systolic Array State (Cycle {cycle})")
        
        # 绘制PE
        for i in range(self.rows):
            for j in range(self.cols):
                pe = self.array[i][j]
                color = 'lightblue' if pe.busy else 'white'
                rect = plt.Rectangle((j, self.rows-i-1), 0.8, 0.8, fill=True, color=color, alpha=0.7)
                plt.gca().add_patch(rect)
                plt.text(j+0.4, self.rows-i-1+0.5, f"{pe.output}", ha='center', va='center')
        
        plt.xlim(0, self.cols)
        plt.ylim(0, self.rows)
        plt.grid(True)
        plt.xticks(np.arange(0.4, self.cols, 1), np.arange(0, self.cols))
        plt.yticks(np.arange(0.4, self.rows, 1), np.arange(0, self.rows)[::-1])
        plt.tight_layout()
        plt.savefig(f"systolic_array_cycle_{cycle}.png")
        plt.close()


def vector_addition_example():
    """向量加法示例"""
    # 创建一个4x4的脉动阵列
    array_size = 4
    systolic = SystolicArray(array_size, array_size)
    
    # 准备输入数据 (两个向量)
    vector_a = [1, 2, 3, 4]
    vector_b = [5, 6, 7, 8]
    
    # 设置输入格式
    horizontal_inputs = [[0 for _ in range(array_size)] for _ in range(array_size)]
    vertical_inputs = [[0 for _ in range(array_size)] for _ in range(array_size)]
    
    # 将向量A放入第一行
    horizontal_inputs[0] = vector_a
    
    # 将向量B放入第一列
    for i in range(array_size):
        vertical_inputs[i][0] = vector_b[i]
    
    # 设置输入
    systolic.set_inputs(horizontal_inputs, vertical_inputs)
    
    # 运行模拟 (需要足够的周期让数据完全流过阵列)
    num_cycles = 2 * array_size
    systolic.run_simulation(num_cycles)
    
    # 获取结果
    results = systolic.outputs
    
    # 打印结果
    print("向量加法结果:")
    for i in range(array_size):
        print(f"结果[{i}] = {results[i][i]}")
    
    # 打印性能指标
    metrics = systolic.get_performance_metrics()
    print("\n性能评估:")
    print(f"能耗: {metrics['energy_pJ']:.2f} pJ")
    print(f"面积: {metrics['area_mm2']:.6f} mm²")
    print(f"延迟: {metrics['latency_cycles']} cycles ({metrics['latency_ns']:.2f} ns)")
    print(f"能效: {metrics['energy_efficiency_TOPS_per_W']:.6f} TOPS/W")
    
    # 可视化
    for cycle in range(num_cycles):
        systolic.visualize_array(cycle)


def matrix_addition_example():
    """矩阵加法示例"""
    # 创建一个4x4的脉动阵列
    array_size = 4
    systolic = SystolicArray(array_size, array_size)
    
    # 准备输入数据 (两个矩阵)
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
    
    # 设置输入格式 (对角线输入方式)
    horizontal_inputs = [[0 for _ in range(array_size*2-1)] for _ in range(array_size)]
    vertical_inputs = [[0 for _ in range(array_size)] for _ in range(array_size*2-1)]
    
    # 将矩阵A沿对角线输入
    for i in range(array_size):
        for j in range(array_size):
            horizontal_inputs[i][i+j] = matrix_a[i][j]
    
    # 将矩阵B沿对角线输入
    for i in range(array_size):
        for j in range(array_size):
            vertical_inputs[i+j][j] = matrix_b[i][j]
    
    # 设置输入
    systolic.set_inputs(horizontal_inputs, vertical_inputs)
    
    # 运行模拟 (需要足够的周期让数据完全流过阵列)
    num_cycles = 2 * array_size
    systolic.run_simulation(num_cycles)
    
    # 获取结果
    results = systolic.outputs
    
    # 打印结果
    print("矩阵加法结果:")
    result_matrix = np.zeros((array_size, array_size))
    for i in range(array_size):
        for j in range(array_size):
            result_matrix[i][j] = results[i][j]
    print(result_matrix)
    
    # 验证结果
    expected = matrix_a + matrix_b
    print("\n预期结果:")
    print(expected)
    
    # 打印性能指标
    metrics = systolic.get_performance_metrics()
    print("\n性能评估:")
    print(f"能耗: {metrics['energy_pJ']:.2f} pJ")
    print(f"面积: {metrics['area_mm2']:.6f} mm²")
    print(f"延迟: {metrics['latency_cycles']} cycles ({metrics['latency_ns']:.2f} ns)")
    print(f"能效: {metrics['energy_efficiency_TOPS_per_W']:.6f} TOPS/W")
    
    # 可视化
    for cycle in range(num_cycles):
        systolic.visualize_array(cycle)


def custom_addition_example(rows, cols, bit_width, visualize=True):
    """
    自定义加法示例
    
    参数:
        rows: 脉动阵列行数
        cols: 脉动阵列列数
        bit_width: 数据位宽
        visualize: 是否可视化
    """
    # 创建脉动阵列
    systolic = SystolicArray(rows, cols, bit_width)
    
    # 准备随机输入数据
    matrix_a = np.random.randint(0, 100, size=(rows, cols))
    matrix_b = np.random.randint(0, 100, size=(rows, cols))
    
    # 设置输入格式 (对角线输入方式)
    horizontal_inputs = [[0 for _ in range(rows+cols-1)] for _ in range(rows)]
    vertical_inputs = [[0 for _ in range(cols)] for _ in range(rows+cols-1)]
    
    # 将矩阵A沿对角线输入
    for i in range(rows):
        for j in range(cols):
            horizontal_inputs[i][i+j] = matrix_a[i][j]
    
    # 将矩阵B沿对角线输入
    for i in range(rows):
        for j in range(cols):
            vertical_inputs[i+j][j] = matrix_b[i][j]
    
    # 设置输入
    systolic.set_inputs(horizontal_inputs, vertical_inputs)
    
    # 运行模拟 (需要足够的周期让数据完全流过阵列)
    num_cycles = rows + cols
    start_time = time.time()
    systolic.run_simulation(num_cycles)
    simulation_time = time.time() - start_time
    
    # 获取结果
    results = systolic.outputs
    
    # 打印结果
    print(f"\n{rows}x{cols} ({bit_width}位) 矩阵加法结果:")
    result_matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result_matrix[i][j] = results[i][j]
    
    # 验证结果
    expected = matrix_a + matrix_b
    is_correct = np.array_equal(result_matrix, expected)
    print(f"结果正确: {is_correct}")
    
    if not is_correct:
        print("结果矩阵:")
        print(result_matrix)
        print("\n预期结果:")
        print(expected)
    
    # 打印性能指标
    metrics = systolic.get_performance_metrics()
    print("\n性能评估:")
    print(f"能耗: {metrics['energy_pJ']:.2f} pJ")
    print(f"面积: {metrics['area_mm2']:.6f} mm²")
    print(f"延迟: {metrics['latency_cycles']} cycles ({metrics['latency_ns']:.2f} ns)")
    print(f"能效: {metrics['energy_efficiency_TOPS_per_W']:.6f} TOPS/W")
    print(f"模拟时间: {simulation_time:.6f} 秒")
    
    # 可视化
    if visualize:
        for cycle in range(min(num_cycles, 10)):  # 只可视化前10个周期
            systolic.visualize_array(cycle)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脉动阵列加法器网络模拟器')
    parser.add_argument('--rows', type=int, default=4, help='脉动阵列行数')
    parser.add_argument('--cols', type=int, default=4, help='脉动阵列列数')
    parser.add_argument('--bit_width', type=int, default=16, help='数据位宽')
    parser.add_argument('--example', type=str, default='matrix', choices=['vector', 'matrix', 'custom'], help='示例类型')
    parser.add_argument('--visualize', action='store_true', help='是否可视化')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("脉动阵列加法器网络模拟器")
    print("=" * 50)
    
    if args.example == 'vector':
        vector_addition_example()
    elif args.example == 'matrix':
        matrix_addition_example()
    else:
        custom_addition_example(args.rows, args.cols, args.bit_width, args.visualize)


if __name__ == "__main__":
    main()

