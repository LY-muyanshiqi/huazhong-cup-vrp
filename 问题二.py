"""
华中杯数学建模A题 - 问题2：环保政策影响下的车辆调度
在问题1基础上增加绿色配送区动态识别和燃油车限行约束
"""

import numpy as np
import pandas as pd
import random
import time
from typing import List, Tuple, Dict, Set
import os
from 问题一_优化版 import (
    Config, DataLoader, TimeVaryingSpeed, Individual, CostCalculator,
    ConstraintValidator, GeneticAlgorithm, ResultWriter
)

# ============== 扩展配置 ==============
class PolicyConfig:
    """政策配置"""
    # 绿色配送区参数
    GREEN_ZONE_CENTER = (0, 0)  # 圆心坐标
    GREEN_ZONE_RADIUS = 10  # 半径(km) - 按题目要求设置为10km

    # 燃油车限行时段
    FUEL_RESTRICTION_START = 8  # 8:00
    FUEL_RESTRICTION_END = 16   # 16:00

# ============== 绿色配送区识别器 ==============
class GreenZoneIdentifier:
    """绿色配送区动态识别器"""

    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = center
        self.radius = radius

    def is_in_green_zone(self, x: float, y: float) -> bool:
        """判断坐标是否在绿色配送区内"""
        distance = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        return distance <= self.radius

    def identify_green_zone_customers(self, coords: pd.DataFrame) -> Set[int]:
        """
        识别绿色配送区内的客户

        Args:
            coords: 客户坐标数据，包含客户ID和坐标列

        Returns:
            绿色配送区内客户ID集合
        """
        green_customers = set()

        # 检测坐标列名
        x_col = None
        y_col = None
        id_col = None
        
        for col in coords.columns:
            col_lower = str(col).lower()
            if 'x' in col_lower:
                x_col = col
            elif 'y' in col_lower:
                y_col = col
            elif 'id' in col_lower or '编号' in col_lower:
                id_col = col
        
        # 如果没有找到合适的列，返回空集合
        if not all([x_col, y_col, id_col]):
            return set()

        for _, row in coords.iterrows():
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                if self.is_in_green_zone(x, y):
                    green_customers.add(int(row[id_col]))
            except (ValueError, TypeError):
                continue

        return green_customers

    def get_green_zone_stats(self, coords: pd.DataFrame) -> Dict:
        """获取绿色配送区统计信息"""
        green_customers = self.identify_green_zone_customers(coords)

        return {
            'total_customers': len(coords),
            'green_zone_customers': len(green_customers),
            'green_zone_rate': len(green_customers) / len(coords) * 100,
            'customer_ids': green_customers
        }

# ============== 时间窗约束分析器 ==============
class TimeWindowAnalyzer:
    """时间窗约束分析器"""

    @staticmethod
    def parse_time_to_hours(time_str) -> float:
        """将时间字符串（如'11:33'）转换为小时数（如11.55）"""
        if isinstance(time_str, (int, float)):
            return float(time_str)
        try:
            # 尝试直接转换为浮点数
            return float(time_str)
        except (ValueError, TypeError):
            # 解析时间字符串格式 'HH:MM'
            if isinstance(time_str, str) and ':' in time_str:
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                return hours + minutes / 60.0
        return 0.0

    @staticmethod
    def requires_restricted_delivery(tw_start: float, tw_end: float) -> bool:
        """
        判断客户是否需要在限行时段配送

        Args:
            tw_start: 时间窗开始时间（小时）
            tw_end: 时间窗结束时间（小时）

        Returns:
            True 如果客户时间窗覆盖8:00-16:00
        """
        return not (tw_end <= PolicyConfig.FUEL_RESTRICTION_START or
                   tw_start >= PolicyConfig.FUEL_RESTRICTION_END)

    @staticmethod
    def get_restricted_customers(time_windows: np.ndarray,
                                 customer_ids: np.ndarray) -> Set[int]:
        """获取需要受限配送的客户"""
        restricted = set()

        for i, tw_data in enumerate(time_windows):
            try:
                # 时间窗数据可能是 [客户编号, 开始时间, 结束时间] 或 [开始时间, 结束时间]
                # 根据数据格式调整索引
                if len(tw_data) >= 3:
                    # 第一列是客户编号，跳过
                    tw_start = TimeWindowAnalyzer.parse_time_to_hours(tw_data[1])
                    tw_end = TimeWindowAnalyzer.parse_time_to_hours(tw_data[2])
                else:
                    tw_start = TimeWindowAnalyzer.parse_time_to_hours(tw_data[0])
                    tw_end = TimeWindowAnalyzer.parse_time_to_hours(tw_data[1])
                    
                if TimeWindowAnalyzer.requires_restricted_delivery(tw_start, tw_end):
                    restricted.add(customer_ids[i])
            except (ValueError, IndexError, TypeError) as e:
                continue

        return restricted

# ============== 车型分配器 ==============
class VehicleAllocator:
    """车型分配器（考虑政策约束）"""

    def __init__(self, green_zone_identifier: GreenZoneIdentifier,
                 time_windows: np.ndarray, customer_ids: np.ndarray,
                 coords: pd.DataFrame = None):
        self.green_zone = green_zone_identifier
        self.time_windows = time_windows
        self.customer_ids = customer_ids
        self.coords = coords

        # 识别受限客户
        if coords is not None:
            self.green_zone_customers = green_zone_identifier.identify_green_zone_customers(coords)
        else:
            self.green_zone_customers = set()
        
        self.restricted_customers = TimeWindowAnalyzer.get_restricted_customers(
            time_windows, customer_ids
        )

        # 必须使用新能源车的客户：同时满足以下条件：
        # 1. 在绿色配送区内
        # 2. 时间窗覆盖限行时段(8:00-16:00)
        self.ev_required = self.green_zone_customers & self.restricted_customers

    def can_use_fuel_vehicle(self, customer_id: int,
                            delivery_time: float = 10.0) -> bool:
        """
        判断某客户是否可以使用燃油车

        Args:
            customer_id: 客户ID
            delivery_time: 预计配送时间
        """
        # 如果客户在绿色配送区
        if customer_id in self.green_zone_customers:
            # 检查时间是否在限行时段
            hour = delivery_time % 24
            if PolicyConfig.FUEL_RESTRICTION_START <= hour < PolicyConfig.FUEL_RESTRICTION_END:
                return False

        return True

    def get_vehicle_type_for_customer(self, customer_id: int) -> str:
        """
        获取适合客户需求的车型

        Returns:
            '新能源1', '新能源2', 或其他可用车型
        """
        if customer_id in self.ev_required:
            # 必须使用新能源车，选择载重大的
            return '新能源1'
        return '燃油车1'

# ============== 问题2遗传算法 ==============
class PolicyGeneticAlgorithm(GeneticAlgorithm):
    """考虑政策约束的遗传算法"""

    def __init__(self, distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                 time_windows: np.ndarray, coords: pd.DataFrame,
                 vehicle_type: str = '燃油车1'):
        super().__init__(distance_matrix, customer_data, time_windows, vehicle_type)

        # 初始化政策组件
        self.coords = coords
        self.green_zone_identifier = GreenZoneIdentifier(
            PolicyConfig.GREEN_ZONE_CENTER,
            PolicyConfig.GREEN_ZONE_RADIUS
        )
        self.vehicle_allocator = VehicleAllocator(
            self.green_zone_identifier,
            time_windows,
            self.customer_ids,
            coords
        )

        # 统计绿色配送区客户
        green_stats = self.green_zone_identifier.get_green_zone_stats(coords)
        print(f"\n绿色配送区信息:")
        print(f"  - 绿色配送区客户数: {green_stats['green_zone_customers']}")
        print(f"  - 占比: {green_stats['green_zone_rate']:.1f}%")
        print(f"  - 必须使用新能源车的客户数: {len(self.vehicle_allocator.ev_required)}")

    def _greedy_construct(self, customers: List[int]) -> List[List[int]]:
        """贪心构造初始解（考虑政策约束）"""
        routes = []
        oversized_customers = {}  # 跟踪需要拆分的客户及其服务次数

        # 首先识别需要拆分的客户
        for customer in customers:
            if customer in self.id_to_idx:
                idx = self.id_to_idx[customer]
                w = self.weights[idx]
                v = self.volumes[idx]
                
                max_weight = Config.VEHICLE_TYPES['燃油车1']['capacity_kg']
                max_volume = Config.VEHICLE_TYPES['燃油车1']['capacity_m3']
                
                # 检查单个客户需求是否超过车辆容量
                if w > max_weight * 0.95 or v > max_volume * 0.95:
                    if customer not in oversized_customers:
                        n_vehicles_needed = max(
                            int(np.ceil(w / (max_weight * 0.95))),
                            int(np.ceil(v / (max_volume * 0.95)))
                        )
                        oversized_customers[customer] = n_vehicles_needed

        # 处理不需要拆分的客户（优先填充车辆）
        processed_customers = set()
        route = []
        current_weight = 0
        current_volume = 0
        use_ev = False
        
        for customer in customers:
            if customer in oversized_customers or customer in processed_customers:
                continue
                
            if customer in self.id_to_idx:
                idx = self.id_to_idx[customer]
                w = self.weights[idx]
                v = self.volumes[idx]

                # 检查是否需要使用新能源车
                if customer in self.vehicle_allocator.ev_required:
                    use_ev = True
                    max_weight = Config.VEHICLE_TYPES['新能源1']['capacity_kg']
                    max_volume = Config.VEHICLE_TYPES['新能源1']['capacity_m3']
                else:
                    max_weight = Config.VEHICLE_TYPES['燃油车1']['capacity_kg']
                    max_volume = Config.VEHICLE_TYPES['燃油车1']['capacity_m3']

                if current_weight + w <= max_weight * 0.95 and \
                   current_volume + v <= max_volume * 0.95:
                    route.append(customer)
                    current_weight += w
                    current_volume += v
                    processed_customers.add(customer)
                else:
                    if route:
                        routes.append(route)
                    route = [customer]
                    current_weight = w
                    current_volume = v
                    use_ev = customer in self.vehicle_allocator.ev_required
                    processed_customers.add(customer)

        if route:
            routes.append(route)

        # 处理需要拆分的客户
        all_splits = []
        for customer, n_vehicles in oversized_customers.items():
            idx = self.id_to_idx[customer]
            w = self.weights[idx]
            v = self.volumes[idx]
            portion_w = w / n_vehicles
            portion_v = v / n_vehicles
            for _ in range(n_vehicles):
                all_splits.append((customer, portion_w, portion_v))
        
        # 贪心装车
        current_route = []
        current_weight = 0
        current_volume = 0
        
        for customer, w, v in all_splits:
            if customer in self.vehicle_allocator.ev_required:
                max_weight = Config.VEHICLE_TYPES['新能源1']['capacity_kg']
                max_volume = Config.VEHICLE_TYPES['新能源1']['capacity_m3']
            else:
                max_weight = Config.VEHICLE_TYPES['燃油车1']['capacity_kg']
                max_volume = Config.VEHICLE_TYPES['燃油车1']['capacity_m3']
            
            if current_weight + w <= max_weight * 0.95 and \
               current_volume + v <= max_volume * 0.95:
                current_route.append(customer)
                current_weight += w
                current_volume += v
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_weight = w
                current_volume = v
        
        if current_route:
            routes.append(current_route)

        return routes if routes else [[]]

    def _regroup_customers(self, customers: List[int]) -> List[List[int]]:
        """重新分组客户到路径（考虑政策约束）- 每个客户只被服务一次"""
        from collections import Counter

        # 每个客户只计算一次真实需求，不管在列表中出现多少次
        customer_total_demand = {}  # {客户ID: (重量, 体积)}

        for cid in customers:
            if cid in self.id_to_idx:
                idx = self.id_to_idx[cid]
                if cid not in customer_total_demand:
                    customer_total_demand[cid] = (self.weights[idx], self.volumes[idx])

        # 分离超重客户和普通客户
        split_customers = []  # [(客户ID, 重量, 体积, 需要的车数)]
        normal_customers = []  # [(客户ID, 重量, 体积)]

        fuel_capacity_weight = Config.VEHICLE_TYPES['燃油车1']['capacity_kg'] * 0.95
        fuel_capacity_volume = Config.VEHICLE_TYPES['燃油车1']['capacity_m3'] * 0.95
        ev_capacity_weight = Config.VEHICLE_TYPES['新能源1']['capacity_kg'] * 0.95
        ev_capacity_volume = Config.VEHICLE_TYPES['新能源1']['capacity_m3'] * 0.95

        for cid, (w, v) in customer_total_demand.items():
            # 根据是否为EV需求客户选择容量约束
            if cid in self.vehicle_allocator.ev_required:
                if w > ev_capacity_weight or v > ev_capacity_volume:
                    n_from_weight = int(np.ceil(w / ev_capacity_weight))
                    n_from_volume = int(np.ceil(v / ev_capacity_volume))
                    n_vehicles = max(n_from_weight, n_from_volume)
                    split_customers.append((cid, w, v, n_vehicles, True))  # True表示需要EV
                else:
                    normal_customers.append((cid, w, v, True))
            else:
                if w > fuel_capacity_weight or v > fuel_capacity_volume:
                    n_from_weight = int(np.ceil(w / fuel_capacity_weight))
                    n_from_volume = int(np.ceil(v / fuel_capacity_volume))
                    n_vehicles = max(n_from_weight, n_from_volume)
                    split_customers.append((cid, w, v, n_vehicles, False))
                else:
                    normal_customers.append((cid, w, v, False))

        # 车辆装载列表 [(当前重量, 当前体积, [客户列表], 需要EV)]
        vehicle_loads = []

        # 先为超重客户分配专用车辆
        for cid, w, v, n_vehicles, need_ev in split_customers:
            portion_w = w / n_vehicles
            portion_v = v / n_vehicles
            for _ in range(n_vehicles):
                vehicle_loads.append((portion_w, portion_v, [cid], need_ev))

        # 按重量降序排列普通客户以便装箱
        normal_customers.sort(key=lambda x: x[1], reverse=True)

        # 将普通客户打包到现有车辆
        for cid, w, v, need_ev in normal_customers:
            capacity_w = ev_capacity_weight if need_ev else fuel_capacity_weight
            capacity_v = ev_capacity_volume if need_ev else fuel_capacity_volume

            placed = False
            for i, (load_w, load_v, route_custs, route_ev) in enumerate(vehicle_loads):
                # 不能混用EV和非EV客户
                if route_ev != need_ev:
                    continue
                if load_w + w <= capacity_w and load_v + v <= capacity_v:
                    vehicle_loads[i] = (load_w + w, load_v + v, route_custs + [cid], need_ev)
                    placed = True
                    break
            if not placed:
                vehicle_loads.append((w, v, [cid], need_ev))

        # 构建路线计划（不考虑EV标志，只保留客户列表）
        route_plan = [load[2] for load in vehicle_loads if load[2]]
        return route_plan if route_plan else [[]]

# ============== 问题2结果输出 ==============
class PolicyResultWriter(ResultWriter):
    """政策影响下的结果输出器"""

    def __init__(self, individual: Individual, cost_calculator: CostCalculator,
                 customer_data: pd.DataFrame, time_windows: np.ndarray,
                 green_stats: Dict, vehicle_allocator: VehicleAllocator):
        super().__init__(individual, cost_calculator, customer_data, time_windows)
        self.green_stats = green_stats
        self.vehicle_allocator = vehicle_allocator

    def save_to_file(self, filename='问题2结果_优化.txt'):
        """保存结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("华中杯数学建模A题 - 问题2优化结果\n")
            f.write("环保政策影响下的车辆调度\n")
            f.write("=" * 70 + "\n\n")

            # 写入政策背景
            f.write("【政策背景】\n")
            f.write("-" * 50 + "\n")
            f.write(f"绿色配送区: 圆心({PolicyConfig.GREEN_ZONE_CENTER[0]}, "
                   f"{PolicyConfig.GREEN_ZONE_CENTER[1]}), 半径{PolicyConfig.GREEN_ZONE_RADIUS}km\n")
            f.write(f"燃油车限行时段: {PolicyConfig.FUEL_RESTRICTION_START}:00-"
                   f"{PolicyConfig.FUEL_RESTRICTION_END}:00\n")
            f.write(f"绿色配送区客户数: {self.green_stats['green_zone_customers']}\n")
            f.write(f"必须使用新能源车客户数: {len(self.vehicle_allocator.ev_required)}\n\n")

            # 写入车辆使用方案
            f.write("【车辆使用方案】\n")
            f.write("-" * 50 + "\n")

            fuel_vehicles = 0
            electric_vehicles = 0

            for i, route in enumerate(self.individual.route_plan, 1):
                # 判断该路线需要使用什么类型的车
                route_ev_required = any(
                    c in self.vehicle_allocator.ev_required
                    for c in route
                )

                vehicle_type = "新能源车" if route_ev_required else "燃油车"
                if route_ev_required:
                    electric_vehicles += 1
                else:
                    fuel_vehicles += 1

                f.write(f"车辆 {i} ({vehicle_type}):\n")
                f.write(f"  配送路径: 配送中心")
                for c in route:
                    marker = "★" if c in self.vehicle_allocator.ev_required else ""
                    f.write(f" → 客户{c}{marker}")
                f.write(" → 配送中心\n")

            f.write(f"\n车辆结构统计:\n")
            f.write(f"  燃油车: {fuel_vehicles}辆\n")
            f.write(f"  新能源车: {electric_vehicles}辆\n\n")

            # 成本计算
            # 检测重量和体积列名
            weight_col = '重量' if '重量' in self.customer_data.columns else '总重量'
            volume_col = '体积' if '体积' in self.customer_data.columns else '总体积'
            id_col = '客户编号' if '客户编号' in self.customer_data.columns else '客户ID'
            
            total_cost, total_carbon, cost_details = self.cost_calculator.calculate_cost(
                self.individual,
                self.customer_data[weight_col].values,
                self.customer_data[volume_col].values,
                self.customer_data[id_col].values
            )

            # 写入成本明细
            f.write("【成本明细】\n")
            f.write("-" * 50 + "\n")
            f.write(f"  启动成本: {cost_details['start_cost']:.2f}元\n")
            f.write(f"  运输成本: {cost_details['transport_cost']:.2f}元\n")
            f.write(f"  等待成本: {cost_details['wait_cost']:.2f}元\n")
            f.write(f"  晚到惩罚: {cost_details['late_penalty']:.2f}元\n")
            f.write(f"  碳排放成本: {cost_details['carbon_cost']:.2f}元\n")
            f.write(f"  --------------------------------\n")
            f.write(f"  总成本: {total_cost:.2f}元\n\n")

            f.write(f"总碳排放: {total_carbon:.2f}kg CO2\n")
            f.write(f"\n" + "=" * 70 + "\n")

        print(f"\n结果已保存到: {filename}")
        return total_cost, total_carbon

# ============== 政策影响分析器 ==============
class PolicyImpactAnalyzer:
    """政策影响分析器"""

    def __init__(self, result1: Dict, result2: Dict):
        self.result1 = result1  # 问题1结果
        self.result2 = result2  # 问题2结果

    def analyze(self) -> Dict:
        """分析政策影响"""
        analysis = {}

        # 成本变化
        cost_change = self.result2['total_cost'] - self.result1['total_cost']
        cost_change_rate = cost_change / self.result1['total_cost'] * 100

        analysis['cost'] = {
            'problem1_cost': self.result1['total_cost'],
            'problem2_cost': self.result2['total_cost'],
            'cost_increase': cost_change,
            'cost_increase_rate': cost_change_rate
        }

        # 碳排放变化
        carbon_change = self.result2['total_carbon'] - self.result1['total_carbon']
        carbon_change_rate = carbon_change / self.result1['total_carbon'] * 100

        analysis['carbon'] = {
            'problem1_carbon': self.result1['total_carbon'],
            'problem2_carbon': self.result2['total_carbon'],
            'carbon_reduction': -carbon_change,
            'carbon_reduction_rate': -carbon_change_rate
        }

        # 车辆结构变化
        analysis['vehicle_structure'] = {
            'problem1_fuel': self.result1.get('fuel_vehicles', 0),
            'problem1_electric': self.result1.get('electric_vehicles', 0),
            'problem2_fuel': self.result2.get('fuel_vehicles', 0),
            'problem2_electric': self.result2.get('electric_vehicles', 0)
        }

        return analysis

    def print_report(self):
        """打印政策影响报告"""
        analysis = self.analyze()

        print("\n" + "=" * 70)
        print("政策影响分析报告")
        print("=" * 70)

        print("\n【成本影响】")
        print(f"  问题1总成本: {analysis['cost']['problem1_cost']:.2f}元")
        print(f"  问题2总成本: {analysis['cost']['problem2_cost']:.2f}元")
        print(f"  成本增加: {analysis['cost']['cost_increase']:.2f}元 "
              f"({analysis['cost']['cost_increase_rate']:+.1f}%)")

        print("\n【碳排放影响】")
        print(f"  问题1碳排放: {analysis['carbon']['problem1_carbon']:.2f}kg CO2")
        print(f"  问题2碳排放: {analysis['carbon']['problem2_carbon']:.2f}kg CO2")
        print(f"  碳减排: {analysis['carbon']['carbon_reduction']:.2f}kg CO2 "
              f"({analysis['carbon']['carbon_reduction_rate']:+.1f}%)")

        print("\n【车辆结构变化】")
        vs = analysis['vehicle_structure']
        print(f"  问题1: 燃油车{vs['problem1_fuel']}辆, 新能源车{vs['problem1_electric']}辆")
        print(f"  问题2: 燃油车{vs['problem2_fuel']}辆, 新能源车{vs['problem2_electric']}辆")

        print("\n【结论】")
        if analysis['cost']['cost_increase_rate'] > 0:
            print(f"  政策导致成本上升{analysis['cost']['cost_increase_rate']:.1f}%，")
            print(f"  但实现了碳减排{analysis['carbon']['carbon_reduction']:.2f}kg CO2。")
        else:
            print("  政策对成本影响较小，且有效降低了碳排放。")

        print("=" * 70)

# ============== 主函数 ==============
def main():
    """主函数"""
    print("=" * 60)
    print("华中杯数学建模A题 - 问题2")
    print("环保政策影响下的车辆调度")
    print("=" * 60)

    start_time = time.time()

    # 加载数据
    print("\n[1/5] 加载数据...")
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()

    n_customers = len(customer_data)
    print(f"客户数量: {n_customers}")

    # 数据预处理
    print("\n[2/5] 数据预处理...")
    # 确保时间窗是numpy数组并转换为浮点数
    if isinstance(time_windows, pd.DataFrame):
        # 转换时间字符串为浮点数（如 "11:33" -> 11.55）
        def time_to_float(time_str):
            try:
                if isinstance(time_str, str) and ':' in time_str:
                    h, m = map(int, time_str.split(':'))
                    return h + m / 60
                elif isinstance(time_str, (int, float)):
                    return float(time_str)
                else:
                    return 8.0  # 默认开始时间
            except:
                return 8.0  # 错误处理
        
        # 应用时间转换
        time_windows = time_windows.values
        converted_time_windows = []
        for row in time_windows:
            if len(row) >= 2:
                start = time_to_float(row[1])
                end = time_to_float(row[2])
                converted_time_windows.append([start, end])
            else:
                converted_time_windows.append([8.0, 18.0])  # 默认时间窗
        time_windows = np.array(converted_time_windows)

    # 确保客户ID列存在且为整数
    if '客户编号' in customer_data.columns:
        customer_data['客户编号'] = customer_data['客户编号'].astype(int)
    elif '客户ID' in customer_data.columns:
        customer_data['客户编号'] = customer_data['客户ID'].astype(int)

    # 识别绿色配送区
    print("\n[3/5] 识别绿色配送区客户...")
    green_zone = GreenZoneIdentifier(
        PolicyConfig.GREEN_ZONE_CENTER,
        PolicyConfig.GREEN_ZONE_RADIUS
    )
    green_stats = green_zone.get_green_zone_stats(coords)

    print(f"  - 绿色配送区客户: {green_stats['green_zone_customers']}个")
    print(f"  - 占比: {green_stats['green_zone_rate']:.1f}%")

    # 运行遗传算法
    print("\n[4/5] 运行考虑政策的遗传算法...")
    ga = PolicyGeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        coords=coords,
        vehicle_type='燃油车1'
    )

    best_individual = ga.run()

    # 输出结果
    print("\n[5/5] 保存结果...")
    # 检测重量和体积列名
    weight_col = '重量' if '重量' in customer_data.columns else '总重量'
    volume_col = '体积' if '体积' in customer_data.columns else '总体积'
    
    # 统计新能源车数量
    n_ev = 0
    for route in best_individual.route_plan:
        # 检查路径是否包含必须使用新能源车的客户
        use_ev = any(c in ga.vehicle_allocator.ev_required for c in route)
        if use_ev:
            n_ev += 1
    
    # 计算总成本（使用燃油车1作为默认车型，实际车型分配在结果输出中处理）
    cost_calculator = CostCalculator(
        distance_matrix, customer_data, time_windows, '燃油车1'
    )
    total_cost, total_carbon, _ = cost_calculator.calculate_cost(
        best_individual,
        customer_data[weight_col].values,
        customer_data[volume_col].values,
        customer_data['客户编号'].values
    )
    
    writer = PolicyResultWriter(
        best_individual, cost_calculator, customer_data, time_windows,
        green_stats, ga.vehicle_allocator
    )
    writer.save_to_file()

    # 打印摘要
    print("\n" + "=" * 60)
    print("求解结果摘要")
    print("=" * 60)
    print(f"使用车辆总数: {best_individual.n_vehicles}辆")
    print(f"  - 燃油车: {best_individual.n_vehicles - n_ev}辆")
    print(f"  - 新能源车: {n_ev}辆")
    print(f"总成本: {total_cost:.2f}元")
    print(f"总碳排放: {total_carbon:.2f}kg CO2")
    print("=" * 60)

    elapsed_time = time.time() - start_time
    print(f"\n总求解时间: {elapsed_time:.2f}秒")

    return {
        'individual': best_individual,
        'total_cost': total_cost,
        'total_carbon': total_carbon,
        'green_stats': green_stats,
        'vehicle_allocator': ga.vehicle_allocator
    }

def compare_with_problem1(result1: Dict, result2: Dict):
    """对比问题1和问题2的结果"""
    print("\n" + "=" * 70)
    print("问题1 vs 问题2 对比分析")
    print("=" * 70)

    analyzer = PolicyImpactAnalyzer(result1, result2)
    analyzer.print_report()

if __name__ == "__main__":
    result2 = main()
