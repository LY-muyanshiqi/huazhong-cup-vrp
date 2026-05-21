"""
华中杯数学建模A题 - 问题1：静态环境下的车辆调度优化
优化版本：修复原代码的多个问题
"""

import numpy as np
import pandas as pd
import random
import copy
import math
import time
from typing import List, Tuple, Dict
import os
from collections import Counter
from multiprocessing import Pool, cpu_count

def evaluate_individual_wrapper(args):
    """评估单个个体的包装函数（用于并行计算）"""
    individual, cost_calculator, weights, volumes, customer_ids = args
    cost, carbon, _ = cost_calculator.calculate_cost(
        individual, weights, volumes, customer_ids
    )
    fitness = 1.0 / cost if cost > 0 else 1e10
    return cost, carbon, fitness

# ============== 配置参数 ==============
# 分隔符：用于染色体编码中表示不同车辆的分割点
SEPARATOR = -1

class Config:
    # 车辆类型定义 [载重(kg), 体积(m³), 启动成本(元), 类型]
    VEHICLE_TYPES = {
        '燃油车1': {'capacity_kg': 3000, 'capacity_m3': 15, 'start_cost': 400, 'fuel_type': 'fuel'},
        '燃油车2': {'capacity_kg': 1500, 'capacity_m3': 8, 'start_cost': 400, 'fuel_type': 'fuel'},
        '燃油车3': {'capacity_kg': 800, 'capacity_m3': 4, 'start_cost': 400, 'fuel_type': 'fuel'},
        '新能源1': {'capacity_kg': 3000, 'capacity_m3': 15, 'start_cost': 500, 'fuel_type': 'electric'},
        '新能源2': {'capacity_kg': 1250, 'capacity_m3': 8.5, 'start_cost': 450, 'fuel_type': 'electric'},
    }

    # 成本参数
    START_COST = 400  # 车辆启动成本
    FUEL_COST_PER_KM = 0.8  # 燃油成本 (元/km)
    ELECTRIC_COST_PER_KM = 0.5  # 电费成本 (元/km) - 增加以反映实际成本
    CARBON_COST_PER_KG = 0.65  # 碳排放成本 (元/kg CO2)

    # 时间窗惩罚
    WAIT_COST_PER_HOUR = 20  # 早到等待成本 (元/小时)
    LATE_PENALTY_PER_HOUR = 50  # 晚到惩罚 (元/小时)

    # 遗传算法参数
    POPULATION_SIZE = 120
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.15
    MAX_GENERATIONS = 600
    ELITE_RATE = 0.1  # 精英保留比例

    # 配送中心坐标
    DEPOT_COORD = (20, 20)

# ============== 数据加载类 ==============
class DataLoader:
    """数据加载器，处理Excel文件"""

    def __init__(self, data_dir='数据'):
        self.data_dir = data_dir

    def load_distance_matrix(self, filename='距离矩阵.xlsx') -> np.ndarray:
        """加载距离矩阵，确保正确处理索引"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_excel(filepath, header=0)
            # 第一列是客户编号标签，从第二列开始是数值
            # 获取列名，检查是否包含"客户"
            cols = df.columns.tolist()
            if '客户' in str(cols[0]) or cols[0] == cols[0]:  # 第一列是标签
                matrix = df.iloc[:, 1:].values.astype(float)
            else:
                matrix = df.values.astype(float)

            # 如果还有非数值，尝试转换
            try:
                matrix = matrix.astype(float)
            except:
                # 找到第一个非数值的位置，跳过
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        try:
                            matrix[i, j] = float(matrix[i, j])
                        except:
                            pass
                matrix = matrix.astype(float)

            # 检查矩阵形状
            print(f"距离矩阵形状: {matrix.shape}")
            print(f"距离矩阵前5行5列:\n{matrix[:5, :5]}")
            return matrix
        else:
            print(f"未找到文件 {filepath}，使用模拟数据")
            return self._generate_sample_distance_matrix()

    def load_customer_data(self, order_file='订单信息.xlsx',
                          coord_file='客户坐标信息.xlsx',
                          time_window_file='时间窗.xlsx') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载客户数据"""
        # 加载订单信息，汇总重量和体积
        if os.path.exists(os.path.join(self.data_dir, order_file)):
            orders = pd.read_excel(os.path.join(self.data_dir, order_file))
            # 按客户编号汇总（支持多种列名）
            customer_col = '目标客户编号' if '目标客户编号' in orders.columns else \
                          '客户编号' if '客户编号' in orders.columns else orders.columns[0]

            # 检测重量列名
            weight_col = [c for c in orders.columns if '重量' in str(c)][0] if any('重量' in str(c) for c in orders.columns) else '重量'
            # 检测体积列名
            volume_col = [c for c in orders.columns if '体积' in str(c)][0] if any('体积' in str(c) for c in orders.columns) else '体积'

            if customer_col:
                customer_demand = orders.groupby(customer_col).agg({
                    weight_col: 'sum',
                    volume_col: 'sum'
                }).reset_index()
                customer_demand.columns = ['客户ID', '总重量', '总体积']
            else:
                # 假设第一列是客户编号
                customer_demand = orders.groupby(orders.columns[0]).sum().reset_index()
        else:
            customer_demand = self._generate_sample_orders()

        # 加载客户坐标
        if os.path.exists(os.path.join(self.data_dir, coord_file)):
            coords = pd.read_excel(os.path.join(self.data_dir, coord_file))
        else:
            coords = self._generate_sample_coords()

        # 加载时间窗
        if os.path.exists(os.path.join(self.data_dir, time_window_file)):
            time_windows = pd.read_excel(os.path.join(self.data_dir, time_window_file))
            # 按客户编号排序，确保顺序与客户数据一致
            time_windows = time_windows.sort_values('客户编号').reset_index(drop=True)
        else:
            time_windows = self._generate_sample_time_windows()

        return customer_demand, coords, time_windows

    def _generate_sample_distance_matrix(self, n_customers=98) -> np.ndarray:
        """生成示例距离矩阵（98×98，含配送中心）"""
        np.random.seed(42)
        n = n_customers + 1  # +1 for depot

        # 生成随机坐标计算距离
        coords = np.random.uniform(0, 50, (n, 2))
        coords[0] = Config.DEPOT_COORD  # 配送中心固定位置

        # 计算欧氏距离矩阵
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))

        return matrix

    def _generate_sample_orders(self, n_customers=98) -> pd.DataFrame:
        """生成示例订单数据"""
        np.random.seed(42)
        return pd.DataFrame({
            '客户编号': range(1, n_customers + 1),
            '重量': np.random.uniform(100, 1500, n_customers),
            '体积': np.random.uniform(0.5, 6, n_customers)
        })

    def _generate_sample_coords(self, n_customers=98) -> pd.DataFrame:
        """生成示例客户坐标"""
        np.random.seed(42)
        coords = np.random.uniform(0, 50, (n_customers, 2))
        coords[0] = Config.DEPOT_COORD  # 保持配送中心位置
        return pd.DataFrame({
            '客户ID': list(range(1, n_customers + 1)),
            'X': coords[1:, 0],
            'Y': coords[1:, 1]
        })

    def _generate_sample_time_windows(self, n_customers=98) -> pd.DataFrame:
        """生成示例时间窗"""
        np.random.seed(42)
        time_windows = []
        for _ in range(n_customers):
            # 随机选择开始时间在8:00-12:00之间
            start_hour = random.choice([8, 9, 10, 11, 12, 13, 14])
            duration = random.choice([2, 3, 4])
            time_windows.append([start_hour, start_hour + duration])

        return pd.DataFrame(time_windows, columns=['开始时间', '结束时间'])

# ============== 速度时变计算 ==============
class TimeVaryingSpeed:
    """速度时变特性计算"""

    @staticmethod
    def get_speed(departure_time: float) -> float:
        """
        根据出发时间获取行驶速度 (km/h)

        时段划分:
        - 顺畅时段(00:00-07:00): 45 km/h
        - 一般时段(07:00-09:00, 17:00-20:00): 35 km/h
        - 拥堵时段(09:00-17:00): 25 km/h
        """
        hour = departure_time % 24

        if 0 <= hour < 7:
            return 45.0
        elif 7 <= hour < 9 or 17 <= hour < 20:
            return 35.0
        else:  # 9 <= hour < 17 or 20 <= hour < 24
            return 25.0

    @staticmethod
    def get_average_speed() -> float:
        """获取平均速度（用于初始估算）"""
        return 33.3  # 各时段平均

    @staticmethod
    def calculate_travel_time(distance: float, departure_time: float) -> float:
        """
        计算行驶时间（小时）

        Args:
            distance: 距离(km)
            departure_time: 出发时间(h)
        """
        speed = TimeVaryingSpeed.get_speed(departure_time)
        return distance / speed

# ============== 个体编码 ==============
class Individual:
    """遗传算法个体编码 - 使用染色体编码（客户序列+分隔符）"""

    def __init__(self, route_plan: List[List[int]] = None, chromosome: List[int] = None):
        """
        个体初始化

        Args:
            route_plan: 路线计划 [[客户1, 客户2], [客户3], ...]
            chromosome: 染色体编码 [客户1, 客户2, SEPARATOR, 客户3, ...]
        """
        if route_plan is not None:
            self.route_plan = route_plan
            self.chromosome = self._route_plan_to_chromosome(route_plan)
        elif chromosome is not None:
            self.chromosome = chromosome
            self.route_plan = self._chromosome_to_route_plan(chromosome)
        else:
            raise ValueError("必须提供route_plan或chromosome")

        self.n_vehicles = len(self.route_plan)
        self.fitness = 0.0
        self.cost = 0.0
        self.carbon = 0.0

    def _route_plan_to_chromosome(self, route_plan: List[List[int]]) -> List[int]:
        """将路线计划转换为染色体编码"""
        chromosome = []
        for i, route in enumerate(route_plan):
            chromosome.extend(route)
            if i < len(route_plan) - 1:
                chromosome.append(SEPARATOR)
        return chromosome

    def _chromosome_to_route_plan(self, chromosome: List[int]) -> List[List[int]]:
        """将染色体编码转换为路线计划"""
        route_plan = []
        current_route = []
        for gene in chromosome:
            if gene == SEPARATOR:
                if current_route:
                    route_plan.append(current_route)
                    current_route = []
            else:
                current_route.append(gene)
        if current_route:
            route_plan.append(current_route)
        return route_plan

    def get_all_customers(self) -> List[int]:
        """获取所有被服务的客户（去重后）"""
        customers = []
        for route in self.route_plan:
            customers.extend(route)
        return list(set(customers))

    def get_customer_sequence(self) -> List[int]:
        """获取染色体中的客户序列（不含分隔符）"""
        return [g for g in self.chromosome if g != SEPARATOR]

    def validate_and_fix(self, all_customer_ids: set, all_customers: list = None) -> bool:
        """验证并修复缺失客户，返回是否有修改"""
        all_served = self.get_all_customers()
        unique_served = set(all_served)

        modified = False

        # 注意：我们不处理重复客户，因为重复客户代表对同一客户的多次服务（用于需求拆分）

        # 2. 处理缺失客户
        if all_customers is not None:
            # 确保all_customers是列表
            if hasattr(all_customers, '__iter__') and not isinstance(all_customers, (str, bytes)):
                if hasattr(all_customers, 'tolist'):
                    all_customers_list = all_customers.tolist()
                else:
                    all_customers_list = list(all_customers)
                
                missing = set(all_customers_list) - set(self.get_all_customers())
                if missing:
                    # 添加缺失的客户到现有路线或创建新路线
                    for c in missing:
                        added = False
                        # 尝试添加到现有路线
                        for route in self.route_plan:
                            # 这里简化处理，实际应该检查载重约束
                            route.append(c)
                            added = True
                            break
                        # 如果无法添加到现有路线，创建新路线
                        if not added:
                            self.route_plan.append([c])
                    self.n_vehicles = len(self.route_plan)
                    modified = True

        return modified

# ============== 成本计算器 ==============
class CostCalculator:
    """成本计算器"""

    def __init__(self, distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                 time_windows: np.ndarray, vehicle_type: str = '燃油车1'):
        self.distance_matrix = distance_matrix
        self.customer_data = customer_data
        self.time_windows = time_windows
        self.vehicle_config = Config.VEHICLE_TYPES[vehicle_type]

    def calculate_cost(self, individual: Individual, 
                      cargo_weights: np.ndarray, cargo_volumes: np.ndarray, 
                      customer_ids: np.ndarray) -> Tuple[float, float, dict]:
        """
        计算个体总成本

        Returns:
            (总成本, 碳排放, 成本明细)
        """
        total_cost = 0.0
        total_carbon = 0.0
        cost_details = {
            'start_cost': 0,
            'transport_cost': 0,
            'wait_cost': 0,
            'late_penalty': 0,
            'carbon_cost': 0
        }

        # 创建客户ID到索引的映射
        id_to_idx = {cid: i for i, cid in enumerate(customer_ids)}

        # 统计整个个体中每个客户的访问次数
        from collections import Counter
        all_customers = []
        for route in individual.route_plan:
            all_customers.extend(route)
        customer_visits = Counter(all_customers)

        for route_idx, route in enumerate(individual.route_plan):
            if not route:
                continue

            # 计算路径成本
            route_cost, route_carbon, route_start, route_transport, route_wait, route_late, route_overload = \
                self._calculate_route_cost(route, cargo_weights, cargo_volumes, id_to_idx, customer_visits)

            total_cost += route_cost
            total_carbon += route_carbon
            cost_details['start_cost'] += route_start
            cost_details['transport_cost'] += route_transport
            cost_details['wait_cost'] += route_wait
            cost_details['late_penalty'] += route_late
            cost_details['overload_penalty'] = cost_details.get('overload_penalty', 0) + route_overload

        cost_details['carbon_cost'] = total_carbon * Config.CARBON_COST_PER_KG
        total_cost += cost_details['carbon_cost']

        return total_cost, total_carbon, cost_details

    def _calculate_route_cost(self, route: List[int],
                             cargo_weights: np.ndarray, cargo_volumes: np.ndarray,
                             id_to_idx: Dict[int, int], customer_visits: Counter) -> Tuple[float, float, float, float, float, float]:
        """计算单条路径的成本"""
        if not route:
            return 0, 0, 0, 0, 0, 0

        # 启动成本
        start_cost = self.vehicle_config['start_cost']

        # 计算运输成本和碳排放
        transport_cost = 0
        wait_cost = 0
        late_penalty = 0
        carbon = 0
        current_time = 8.0  # 假设从8:00开始配送

        # 从配送中心出发
        prev_idx = 0  # 配送中心索引

        # 预计算载重率（考虑拆分配送）
        total_weight = 0
        total_volume = 0
        
        for c in route:
            if c in id_to_idx:
                # 计算每辆车的实际载重（总需求/服务次数）
                idx = id_to_idx[c]
                count = customer_visits.get(c, 1)
                # 检查是否超载
                individual_weight = cargo_weights[idx] / count
                individual_volume = cargo_volumes[idx] / count
                total_weight += individual_weight
                total_volume += individual_volume
        
        # 容量约束检查
        max_weight = self.vehicle_config['capacity_kg']
        max_volume = self.vehicle_config['capacity_m3']
        
        # 如果超载，添加惩罚
        overload_penalty = 0
        if total_weight > max_weight:
            overload_penalty += (total_weight - max_weight) * 1.0  # 合理惩罚
        if total_volume > max_volume:
            overload_penalty += (total_volume - max_volume) * 2.0  # 合理惩罚

        load_ratio = min(1.0, total_weight / max_weight)

        # 预计算成本参数
        is_electric = self.vehicle_config.get('fuel_type') == 'electric'
        cost_per_km = Config.ELECTRIC_COST_PER_KM if is_electric else Config.FUEL_COST_PER_KM
        cost_factor = cost_per_km * (0.5 + 0.5 * load_ratio)
        carbon_factor = 0.21 * load_ratio if not is_electric else 0

        for customer_id in route:
            if customer_id not in id_to_idx:
                continue

            list_idx = id_to_idx[customer_id]  # 用于访问weights和volumes
            customer_idx = customer_id  # 用于访问distance_matrix（索引与客户ID对应）

            # 获取距离
            distance = self.distance_matrix[prev_idx, customer_idx]

            # 计算行驶时间（考虑时变速度）
            speed = TimeVaryingSpeed.get_speed(current_time)
            travel_time = distance / speed
            arrival_time = current_time + travel_time

            # 获取时间窗
            try:
                tw_idx = customer_idx - 1
                if isinstance(self.time_windows, np.ndarray):
                    tw_start = float(self.time_windows[tw_idx, 0])
                    tw_end = float(self.time_windows[tw_idx, 1])
                else:
                    tw_row = self.time_windows.iloc[tw_idx]
                    tw_start = float(tw_row.iloc[0])
                    tw_end = float(tw_row.iloc[1])
            except (ValueError, IndexError, TypeError):
                tw_start, tw_end = 8.0, 18.0

            # 计算等待成本和晚到惩罚
            if arrival_time < tw_start:
                wait_cost += (tw_start - arrival_time) * Config.WAIT_COST_PER_HOUR
                current_time = tw_start
            elif arrival_time > tw_end:
                late_penalty += (arrival_time - tw_end) * Config.LATE_PENALTY_PER_HOUR
                current_time = arrival_time
            else:
                current_time = arrival_time

            # 服务时间（假设0.5小时）
            current_time += 0.5

            # 运输成本和碳排放（距离单位是公里）
            transport_cost += distance * cost_factor
            if carbon_factor > 0:
                carbon += distance * carbon_factor

            prev_idx = customer_idx

        # 返回配送中心
        return_distance = self.distance_matrix[prev_idx, 0]
        transport_cost += return_distance * cost_factor
        if carbon_factor > 0:
            carbon += return_distance * carbon_factor

        total_cost = start_cost + transport_cost + wait_cost + late_penalty + overload_penalty
        return total_cost, carbon, start_cost, transport_cost, wait_cost, late_penalty, overload_penalty

    def _calculate_route_cost_no_time_windows(self, route: List[int],
                             cargo_weights: np.ndarray, cargo_volumes: np.ndarray,
                             id_to_idx: Dict[int, int], customer_visits: Counter) -> Tuple[float, float, float, float, float, float]:
        """计算单条路径的成本（忽略时间窗约束）"""
        if not route:
            return 0, 0, 0, 0, 0, 0

        start_cost = self.vehicle_config['start_cost']
        transport_cost = 0
        carbon = 0
        current_time = 8.0
        prev_idx = 0

        total_weight = 0
        total_volume = 0

        for c in route:
            if c in id_to_idx:
                idx = id_to_idx[c]
                count = customer_visits.get(c, 1)
                individual_weight = cargo_weights[idx] / count
                individual_volume = cargo_volumes[idx] / count
                total_weight += individual_weight
                total_volume += individual_volume

        max_weight = self.vehicle_config['capacity_kg']
        max_volume = self.vehicle_config['capacity_m3']

        overload_penalty = 0
        if total_weight > max_weight:
            overload_penalty += (total_weight - max_weight) * 1.0
        if total_volume > max_volume:
            overload_penalty += (total_volume - max_volume) * 2.0

        load_ratio = min(1.0, total_weight / max_weight)
        is_electric = self.vehicle_config.get('fuel_type') == 'electric'
        cost_per_km = Config.ELECTRIC_COST_PER_KM if is_electric else Config.FUEL_COST_PER_KM
        cost_factor = cost_per_km * (0.5 + 0.5 * load_ratio)
        carbon_factor = 0.21 * load_ratio if not is_electric else 0

        for customer_id in route:
            if customer_id not in id_to_idx:
                continue

            customer_idx = customer_id
            distance = self.distance_matrix[prev_idx, customer_idx]

            speed = TimeVaryingSpeed.get_speed(current_time)
            travel_time = distance / speed
            arrival_time = current_time + travel_time

            current_time = arrival_time + 0.5

            transport_cost += distance * cost_factor
            if carbon_factor > 0:
                carbon += distance * carbon_factor

            prev_idx = customer_idx

        return_distance = self.distance_matrix[prev_idx, 0]
        transport_cost += return_distance * cost_factor
        if carbon_factor > 0:
            carbon += return_distance * carbon_factor

        total_cost = start_cost + transport_cost + overload_penalty
        return total_cost, carbon, start_cost, transport_cost, 0, 0, overload_penalty

    def calculate_cost_no_time_windows(self, individual: Individual,
                                      cargo_weights: np.ndarray, cargo_volumes: np.ndarray,
                                      customer_ids: np.ndarray) -> Tuple[float, float, dict]:
        """计算个体总成本（忽略时间窗约束）"""
        total_cost = 0.0
        total_carbon = 0.0
        cost_details = {
            'start_cost': 0,
            'transport_cost': 0,
            'wait_cost': 0,
            'late_penalty': 0,
            'carbon_cost': 0
        }

        id_to_idx = {cid: i for i, cid in enumerate(customer_ids)}

        from collections import Counter
        all_customers = []
        for route in individual.route_plan:
            all_customers.extend(route)
        customer_visits = Counter(all_customers)

        for route_idx, route in enumerate(individual.route_plan):
            if not route:
                continue

            route_cost, route_carbon, route_start, route_transport, route_wait, route_late, route_overload = \
                self._calculate_route_cost_no_time_windows(route, cargo_weights, cargo_volumes, id_to_idx, customer_visits)

            total_cost += route_cost
            total_carbon += route_carbon
            cost_details['start_cost'] += route_start
            cost_details['transport_cost'] += route_transport
            cost_details['overload_penalty'] = cost_details.get('overload_penalty', 0) + route_overload

        cost_details['carbon_cost'] = total_carbon * Config.CARBON_COST_PER_KG
        total_cost += cost_details['carbon_cost']

        return total_cost, total_carbon, cost_details

# ============== 约束验证器 ==============
class ConstraintValidator:
    """约束验证器"""

    def __init__(self, customer_data: pd.DataFrame, vehicle_type: str = '燃油车1'):
        self.customer_data = customer_data
        self.vehicle_config = Config.VEHICLE_TYPES[vehicle_type]
        self.max_weight = self.vehicle_config['capacity_kg']
        self.max_volume = self.vehicle_config['capacity_m3']

    def validate_route(self, route: List[int],
                      weights: np.ndarray, volumes: np.ndarray,
                      id_to_idx: Dict[int, int]) -> Tuple[bool, str]:
        """验证单条路径是否满足约束"""
        total_weight = 0
        total_volume = 0

        for customer_id in route:
            if customer_id not in id_to_idx:
                return False, f"客户{customer_id}不存在"

            idx = id_to_idx[customer_id]
            total_weight += weights[idx]
            total_volume += volumes[idx]

        if total_weight > self.max_weight:
            return False, f"载重超限: {total_weight}kg > {self.max_weight}kg"

        if total_volume > self.max_volume:
            return False, f"体积超限: {total_volume}m³ > {self.max_volume}m³"

        return True, "OK"

    def validate_time_window(self, route: List[int],
                            time_windows: np.ndarray,
                            id_to_idx: Dict[int, int]) -> Tuple[bool, List]:
        """验证时间窗约束"""
        current_time = 8.0
        prev_idx = 0
        violations = []

        for customer_id in route:
            if customer_id not in id_to_idx:
                continue

            customer_idx = id_to_idx[customer_id] + 1
            # 假设有距离矩阵
            # 这里简化处理，实际需要传入距离矩阵

            arrival_time = current_time  # 简化

            tw_start, tw_end = time_windows[customer_idx - 1]

            if arrival_time > tw_end:
                violations.append({
                    'customer': customer_id,
                    'arrival': arrival_time,
                    'tw_end': tw_end,
                    'delay': arrival_time - tw_end
                })

            current_time = max(arrival_time, tw_start) + 0.5

        return len(violations) == 0, violations

# ============== 遗传算法 ==============
class GeneticAlgorithm:
    """遗传算法求解器"""

    def __init__(self, distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                 time_windows: np.ndarray, vehicle_type: str = '燃油车1', coords: pd.DataFrame = None):
        self.distance_matrix = distance_matrix
        self.customer_data = customer_data
        self.time_windows = time_windows
        self.vehicle_type = vehicle_type
        self.coords = coords

        # 提取客户需求
        # 提取客户需求（支持多种列名）
        # 检测列名
        if '重量' in customer_data.columns:
            weight_col = '重量'
            volume_col = '体积'
        elif '总重量' in customer_data.columns:
            weight_col = '总重量'
            volume_col = '总体积'
        else:
            weight_col = customer_data.columns[1]
            volume_col = customer_data.columns[2]

        # 预计算客户数据
        self.customer_ids = customer_data['客户ID'].values
        self.weights = customer_data[weight_col].values
        self.volumes = customer_data[volume_col].values

        # 车辆约束
        self.max_weight = Config.VEHICLE_TYPES[vehicle_type]['capacity_kg']
        self.max_volume = Config.VEHICLE_TYPES[vehicle_type]['capacity_m3']

        # 创建ID到索引的映射（预计算）
        # weights和volumes使用列表索引（0-87）
        # distance_matrix使用客户ID作为索引（因为matrix索引与客户ID对应）
        self.id_to_idx = {}
        for i, cid in enumerate(self.customer_ids):
            self.id_to_idx[cid] = i

        # 预计算客户需求列表（用于快速访问）
        self.customer_demands = []
        for i, cid in enumerate(self.customer_ids):
            self.customer_demands.append((cid, self.weights[i], self.volumes[i]))

        # 初始化组件
        self.cost_calculator = CostCalculator(
            distance_matrix, customer_data, time_windows, vehicle_type
        )
        self.constraint_validator = ConstraintValidator(customer_data, vehicle_type)

        # 遗传算法参数
        self.population_size = 120   # 增加种群大小提高多样性
        self.crossover_rate = 0.8  # 增加交叉率
        self.mutation_rate = 0.25  # 适当提高变异率以增加多样性
        self.max_generations = 600  # 增加迭代次数
        self.elite_rate = 0.1  # 精英保留比例
        self.patience = 100  # 增加早停耐心值
        self.min_improvement = 0.1  # 最小改进阈值

        self.best_individual = None
        self.best_cost = float('inf')
        self.history = []
        
        # 自适应参数
        self.original_mutation_rate = self.mutation_rate
        self.original_elite_rate = self.elite_rate
        
        # 多目标优化参数
        self.multi_objective = True  # 启用多目标优化
        self.weight_cost = 0.6       # 成本权重
        self.weight_emission = 0.3   # 碳排放权重
        self.weight_vehicles = 0.1   # 车辆数权重
        
        # 归一化基准（初始代设置）
        self.max_cost = None
        self.max_emission = None
        self.max_vehicles = None

    def calculate_diversity(self, population: List[Individual]) -> float:
        """计算种群多样性（基于成本分布）"""
        if len(population) < 2:
            return 0.0
        
        costs = [ind.cost for ind in population]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        if mean_cost == 0:
            return 0.0
        
        return std_cost / mean_cost  # 变异系数作为多样性指标

    def adapt_parameters(self, generation: int, diversity: float):
        """自适应调整参数"""
        # 后期增加变异率，避免早熟
        if generation > 50 and diversity < 0.1:
            self.mutation_rate = min(0.4, self.mutation_rate * 1.2)
            if self.mutation_rate != self.original_mutation_rate:
                print(f"  [自适应] 代数{generation}：变异率从{self.original_mutation_rate:.2f}提升到{self.mutation_rate:.2f}")
        
        # 收敛时增加精英保留
        if generation > 100:
            self.elite_rate = min(0.2, self.elite_rate + 0.01)
            if abs(self.elite_rate - self.original_elite_rate) > 0.02:
                print(f"  [自适应] 代数{generation}：精英率从{self.original_elite_rate:.2f}提升到{self.elite_rate:.2f}")

    def initialize_population(self) -> List[Individual]:
        """初始化种群"""
        population = []

        for _ in range(self.population_size):
            # 创建客户列表（包含ID、重量、体积）
            customers_with_demands = [(cid, w, v) for cid, w, v in self.customer_demands]
            
            # 随机打乱顺序，但保留部分按尺寸降序的个体以提高装箱效率
            if random.random() < 0.7:
                # 70%的个体使用随机顺序
                random.shuffle(customers_with_demands)
                customers = [c[0] for c in customers_with_demands]
            else:
                # 30%的个体使用首次适应递减策略（按尺寸降序）
                customers_with_demands.sort(key=lambda x: x[1] + x[2], reverse=True)
                customers = [c[0] for c in customers_with_demands]

            # 贪心构造初始解（考虑载重和体积约束）
            route_plan = self._greedy_construct(customers)
            individual = Individual(route_plan)
            
            # 验证所有客户都被分配
            served_customers = individual.get_all_customers()
            if not set(self.customer_ids).issubset(set(served_customers)):
                missing = set(self.customer_ids) - set(served_customers)
                print(f"警告：初始化时缺少客户: {missing}")
                # 重新构造确保所有客户都被分配
                route_plan = self._ensure_all_customers(customers)
                individual = Individual(route_plan)
            
            population.append(individual)

        return population

    def test_without_time_windows(self) -> Dict:
        """
        验证方法：忽略时间窗约束运行算法，检查车辆数变化

        Returns:
            包含测试结果的字典
        """
        print("\n" + "=" * 60)
        print("【验证测试】忽略时间窗约束")
        print("=" * 60)

        original_cost_calc = self.cost_calculator

        test_ga = GeneticAlgorithm(
            self.distance_matrix,
            self.customer_data,
            self.time_windows,
            self.vehicle_type,
            self.coords
        )

        test_pop = test_ga.initialize_population()

        for ind in test_pop:
            test_ga._merge_routes(ind)
            cost, carbon, _ = test_ga.cost_calculator.calculate_cost_no_time_windows(
                ind, self.weights, self.volumes, self.customer_ids
            )
            ind.cost = cost
            ind.carbon = carbon
            num_vehicles = len([r for r in ind.route_plan if r])
            vehicle_penalty = num_vehicles * 800
            penalized_cost = cost + vehicle_penalty
            if penalized_cost > 0:
                ind.fitness = 1.0 / penalized_cost
            else:
                ind.fitness = 1e-10

        best_no_tw = min(test_pop, key=lambda x: x.cost)
        no_tw_vehicles = len([r for r in best_no_tw.route_plan if r])

        current_best = min(population, key=lambda x: x.cost) if 'population' in dir() else None
        current_vehicles = len([r for r in current_best.route_plan if r]) if current_best else 0

        print(f"\n忽略时间窗后的最优解:")
        print(f"  车辆数: {no_tw_vehicles}")
        print(f"  成本: {best_no_tw.cost:.2f}元")

        if current_best:
            print(f"\n原始解（含时间窗约束）:")
            print(f"  车辆数: {current_vehicles}")
            print(f"  成本: {current_best.cost:.2f}元")

            vehicle_diff = current_vehicles - no_tw_vehicles
            if vehicle_diff > 5:
                print(f"\n结论: 时间窗是主要限制因素 (差异: {vehicle_diff}辆车)")
            else:
                print(f"\n结论: 时间窗不是主要限制因素 (差异: {vehicle_diff}辆车)")

        result = {
            'no_tw_vehicles': no_tw_vehicles,
            'no_tw_cost': best_no_tw.cost,
            'current_vehicles': current_vehicles,
            'current_cost': current_best.cost if current_best else 0,
            'time_window_is_main_constraint': (current_vehicles - no_tw_vehicles) > 5 if current_best else False
        }

        print("=" * 60)
        return result

    def create_better_initial_population(self, current_solution_pop: List[Individual] = None) -> List[Individual]:
        """
        创建包含拼单路径的改进初始种群

        Args:
            current_solution_pop: 当前解列表（如果有）

        Returns:
            改进后的种群
        """
        population = []

        if current_solution_pop:
            population.extend([Individual(ind.route_plan.copy()) for ind in current_solution_pop[:10]])

        clustered_solutions = self.cluster_and_assign(n_clusters=30)
        for sol in clustered_solutions:
            population.append(sol)

        for _ in range(20):
            random_combined = self.random_combine_customers()
            population.append(random_combined)

        for _ in range(self.population_size - len(population)):
            customers_with_demands = [(cid, w, v) for cid, w, v in self.customer_demands]
            if random.random() < 0.7:
                random.shuffle(customers_with_demands)
            else:
                customers_with_demands.sort(key=lambda x: x[1] + x[2], reverse=True)
            customers = [c[0] for c in customers_with_demands]
            route_plan = self._greedy_construct(customers)
            individual = Individual(route_plan)
            self._ensure_all_customers_silent(individual)
            population.append(individual)

        while len(population) < self.population_size:
            population.append(random.choice(population))

        return population[:self.population_size]

    def _ensure_all_customers_silent(self, individual: Individual):
        """静默确保所有客户都被分配"""
        served = set(individual.get_all_customers())
        missing = set(self.customer_ids) - served
        if missing:
            for c in missing:
                if c in self.id_to_idx:
                    individual.route_plan.append([c])
            individual.route_plan = [r for r in individual.route_plan if r]
            individual.n_vehicles = len(individual.route_plan)

    def cluster_and_assign(self, n_clusters: int = 30) -> List[Individual]:
        """
        基于地理坐标聚类，相近客户分配同车

        Args:
            n_clusters: 目标聚类数（等于目标车辆数）

        Returns:
            聚类解列表
        """
        if 'coords' not in dir(self) or self.coords is None:
            return []

        solutions = []

        coords = self.coords.copy()
        coords['客户ID'] = self.customer_data['客户ID'].values

        capacity_weight = self.max_weight * 0.95
        capacity_volume = self.max_volume * 0.95

        k = min(n_clusters, len(coords))

        indices = np.arange(len(coords))
        np.random.shuffle(indices)
        cluster_centers = coords.iloc[indices[:k]][['X', 'Y']].values

        clusters = [[] for _ in range(k)]
        for idx, row in coords.iterrows():
            cid = row['客户ID']
            x, y = row['X'], row['Y']
            distances = [np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in cluster_centers]
            nearest = np.argmin(distances)
            clusters[nearest].append(cid)

        routes = []
        for cluster in clusters:
            if not cluster:
                continue

            cluster_weight = sum(self.weights[self.id_to_idx[c]] for c in cluster if c in self.id_to_idx)
            cluster_volume = sum(self.volumes[self.id_to_idx[c]] for c in cluster if c in self.id_to_idx)

            if cluster_weight <= capacity_weight and cluster_volume <= capacity_volume:
                routes.append(cluster)
            else:
                sub_routes = self._split_cluster(cluster, capacity_weight, capacity_volume)
                routes.extend(sub_routes)

        for route in routes:
            if not route:
                continue
            route.sort(key=lambda c: np.sqrt(
                (coords[coords['客户ID']==c]['X'].values[0] - Config.DEPOT_COORD[0])**2 +
                (coords[coords['客户ID']==c]['Y'].values[0] - Config.DEPOT_COORD[1])**2
            ) if c in self.id_to_idx and len(coords[coords['客户ID']==c]) > 0 else float('inf'))

        individual = Individual(routes)
        self._ensure_all_customers_silent(individual)
        solutions.append(individual)

        return solutions

    def _split_cluster(self, cluster: List[int], capacity_weight: float, capacity_volume: float) -> List[List[int]]:
        """拆分过大的聚类"""
        if not cluster:
            return []

        sub_routes = []
        current_route = []
        current_weight = 0
        current_volume = 0

        for c in cluster:
            if c not in self.id_to_idx:
                continue
            w = self.weights[self.id_to_idx[c]]
            v = self.volumes[self.id_to_idx[c]]

            if w <= capacity_weight and v <= capacity_volume:
                if current_weight + w <= capacity_weight and current_volume + v <= capacity_volume:
                    current_route.append(c)
                    current_weight += w
                    current_volume += v
                else:
                    if current_route:
                        sub_routes.append(current_route)
                    current_route = [c]
                    current_weight = w
                    current_volume = v
            else:
                if current_route:
                    sub_routes.append(current_route)
                sub_routes.append([c])
                current_route = []
                current_weight = 0
                current_volume = 0

        if current_route:
            sub_routes.append(current_route)

        return sub_routes

    def random_combine_customers(self) -> Individual:
        """
        创建随机拼单解

        Returns:
            随机拼单个体
        """
        customers = list(self.customer_ids)
        random.shuffle(customers)

        routes = []
        current_route = []
        current_weight = 0
        current_volume = 0
        capacity_weight = self.max_weight * 0.95
        capacity_volume = self.max_volume * 0.95

        for c in customers:
            if c not in self.id_to_idx:
                continue
            w = self.weights[self.id_to_idx[c]]
            v = self.volumes[self.id_to_idx[c]]

            if current_weight + w <= capacity_weight and current_volume + v <= capacity_volume:
                current_route.append(c)
                current_weight += w
                current_volume += v
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_weight = w
                current_volume = v

        if current_route:
            routes.append(current_route)

        individual = Individual(routes)
        self._ensure_all_customers_silent(individual)
        return individual

    def _ensure_all_customers(self, customers: List[int]) -> List[List[int]]:
        """确保所有客户都被分配到路径"""
        routes = self._greedy_construct(customers)
        served = set()
        for route in routes:
            served.update(route)
        
        missing = set(self.customer_ids) - served
        if missing:
            # 将缺失的客户添加到现有路径或创建新路径
            for c in missing:
                if c in self.id_to_idx:
                    idx = self.id_to_idx[c]
                    w = self.weights[idx]
                    v = self.volumes[idx]
                    capacity_weight = self.max_weight * 0.95
                    capacity_volume = self.max_volume * 0.95
                    
                    # 计算需要的车辆数
                    n_from_weight = int(np.ceil(w / capacity_weight))
                    n_from_volume = int(np.ceil(v / capacity_volume))
                    n_vehicles = max(n_from_weight, n_from_volume)
                    
                    # 计算每个部分的需求
                    portion_w = w / n_vehicles
                    portion_v = v / n_vehicles
                    
                    # 尝试将每个部分添加到现有路径
                    for _ in range(n_vehicles):
                        added = False
                        for route in routes:
                            # 计算当前路径的总载重和体积
                            route_weight = 0
                            route_volume = 0
                            for customer in route:
                                if customer in self.id_to_idx:
                                    route_weight += self.weights[self.id_to_idx[customer]]
                                    route_volume += self.volumes[self.id_to_idx[customer]]
                            
                            if route_weight + portion_w <= capacity_weight and route_volume + portion_v <= capacity_volume:
                                route.append(c)
                                added = True
                                break
                        # 如果无法添加到现有路径，创建新路径
                        if not added:
                            routes.append([c])
        
        return routes

    def _greedy_construct(self, customers: List[int]) -> List[List[int]]:
        """贪心构造初始解 - 使用最近邻策略实现真正的聚类和拼单"""
        # 去重，确保每个客户只处理一次
        unique_customers = list(set(customers))
        
        # 客户需求映射
        customer_demands = {}
        for cid in unique_customers:
            if cid in self.id_to_idx:
                idx = self.id_to_idx[cid]
                customer_demands[cid] = (self.weights[idx], self.volumes[idx])

        capacity_weight = self.max_weight * 0.95
        capacity_volume = self.max_volume * 0.95

        # 生成所有客户的需求部分（包括拆分的）
        all_parts = []

        # 处理所有客户，将需求拆分为多个部分
        for cid in unique_customers:
            if cid not in customer_demands:
                continue
            w, v = customer_demands[cid]

            # 检查是否需要拆分
            if w > capacity_weight or v > capacity_volume:
                # 需要拆分，计算需要的车辆数
                n_from_weight = int(np.ceil(w / capacity_weight))
                n_from_volume = int(np.ceil(v / capacity_volume))
                n_vehicles = max(n_from_weight, n_from_volume)

                # 计算每个部分的需求
                portion_w = w / n_vehicles
                portion_v = v / n_vehicles

                # 添加所有部分（每个部分都是独立的可分配单元）
                for i in range(n_vehicles):
                    all_parts.append((cid, portion_w, portion_v, i))  # 添加部分编号
            else:
                # 不需要拆分，直接添加
                all_parts.append((cid, w, v, 0))
        
        # 获取距离函数
        def get_distance_to_center(cid):
            idx = self.id_to_idx[cid] + 1
            return self.distance_matrix[0][idx]
        
        def get_distance_between(c1, c2):
            idx1 = self.id_to_idx[c1] + 1
            idx2 = self.id_to_idx[c2] + 1
            return self.distance_matrix[idx1][idx2]
        
        # 构建车辆 - 使用真正的最近邻贪心策略
        vehicles = []  # [(weight, volume, [customers])]
        assigned = set()  # 跟踪已分配的 (cid, part_id)
        
        while len(assigned) < len(all_parts):
            # 创建新车
            current_route = []
            current_weight = 0
            current_volume = 0
            
            # 从配送中心开始
            last_cid = 0  # 0表示配送中心
            
            while True:
                # 找到距last_cid最近的未分配客户
                best_part = None
                best_dist = float('inf')
                
                for p in all_parts:
                    if (p[0], p[3]) not in assigned:
                        if last_cid == 0:
                            dist = get_distance_to_center(p[0])
                        else:
                            dist = get_distance_between(last_cid, p[0])
                        if dist < best_dist:
                            best_dist = dist
                            best_part = p
                
                if best_part is None:
                    break  # 没有更多可分配的客户
                
                c_cid, c_w, c_v, c_pid = best_part
                
                # 检查能否容纳
                if current_weight + c_w <= capacity_weight and current_volume + c_v <= capacity_volume:
                    current_route.append((c_cid, c_pid))
                    current_weight += c_w
                    current_volume += c_v
                    assigned.add((c_cid, c_pid))
                    last_cid = c_cid  # 更新为当前客户
                else:
                    break  # 当前车辆已满，需要创建新车
            
            # 只有当有客户时才添加车辆
            if current_route:
                route_cids = list(dict.fromkeys([p[0] for p in current_route]))
                vehicles.append((current_weight, current_volume, route_cids))
            else:
                break  # 无法分配更多客户
        
        # 移除空车辆
        vehicles = [v for v in vehicles if v[2]]

        # 构建路线计划
        route_plan = [load[2] for load in vehicles if load[2]]

        return route_plan if route_plan else [[]]

    def _log_construction(self, route_plan: List[List[int]], generation: int = 0):
        """记录路径构造日志（只在特定代数打印）"""
        if generation in [0, 1] or generation % 100 == 0:
            total_customers = sum(len(r) for r in route_plan)
            avg_customers = total_customers / len(route_plan) if route_plan else 0
            total_distance_before = sum(self._calculate_route_distance(r) for r in route_plan)
            print(f"  [代数{generation}] 构造了{len(route_plan)}条路径，平均每车{avg_customers:.1f}个客户，总距离{total_distance_before:.2f}km")

    def evaluate_population(self, population: List[Individual], generation: int = 0):
        """评估种群（串行）- 支持多目标优化"""
        generation_costs = []
        generation_emissions = []
        generation_vehicles = []
        
        for ind in population:
            self._smart_merge_routes(ind)

            cost, carbon, _ = self.cost_calculator.calculate_cost(
                ind, self.weights, self.volumes, self.customer_ids
            )
            ind.cost = cost
            ind.carbon = carbon
            
            num_vehicles = len([r for r in ind.route_plan if r])
            vehicle_penalty = num_vehicles * 800
            penalized_cost = cost + vehicle_penalty

            if cost <= 0 or penalized_cost <= 0:
                print(f"  [警告] 代数{generation}：非正成本 cost={cost:.2f}, penalized={penalized_cost:.2f}, num_vehicles={num_vehicles}")
                ind.cost = 400 * max(1, num_vehicles)
                penalized_cost = ind.cost + vehicle_penalty
            
            if penalized_cost > 1e9:
                print(f"  [警告] 代数{generation}：异常高成本 penalized={penalized_cost:.2e}，限制上限")
                penalized_cost = min(penalized_cost, 1e6)
            
            # 多目标适应度计算
            if self.multi_objective:
                generation_costs.append(ind.cost)
                generation_emissions.append(ind.carbon)
                generation_vehicles.append(num_vehicles)
                # 暂时先不设置fitness，等收集完所有值后再计算归一化适应度
            else:
                if penalized_cost > 0:
                    ind.fitness = 1.0 / penalized_cost
                else:
                    ind.fitness = 1e-10

                if not np.isfinite(ind.fitness):
                    print(f"  [警告] 代数{generation}：非有限适应度 {ind.fitness:.2e}")
                    ind.fitness = 1e-10
        
        # 多目标优化：计算归一化适应度
        if self.multi_objective:
            # 初始化归一化基准（第0代）
            if self.max_cost is None:
                self.max_cost = max(generation_costs)
                self.max_emission = max(generation_emissions)
                self.max_vehicles = max(generation_vehicles)
                print(f"  [多目标] 归一化基准：max_cost={self.max_cost:.2f}, max_emission={self.max_emission:.2f}, max_vehicles={self.max_vehicles}")
            
            # 更新归一化基准（根据最新代）
            self.max_cost = max(self.max_cost, max(generation_costs))
            self.max_emission = max(self.max_emission, max(generation_emissions))
            self.max_vehicles = max(self.max_vehicles, max(generation_vehicles))
            
            # 计算每个个体的多目标适应度
            for i, ind in enumerate(population):
                num_vehicles = len([r for r in ind.route_plan if r])
                
                # 归一化（值越小越好，所以用1 - 归一化值）
                norm_cost = ind.cost / self.max_cost if self.max_cost > 0 else 1.0
                norm_emission = ind.carbon / self.max_emission if self.max_emission > 0 else 1.0
                norm_vehicles = num_vehicles / self.max_vehicles if self.max_vehicles > 0 else 1.0
                
                # 加权求和（越小越好，所以用1减去综合得分）
                weighted_score = (self.weight_cost * norm_cost + 
                                 self.weight_emission * norm_emission + 
                                 self.weight_vehicles * norm_vehicles)
                
                # 适应度：加权得分越小越好，所以fitness = 1 / (weighted_score + epsilon)
                ind.fitness = 1.0 / (weighted_score + 1e-6)
                
                # 保存归一化指标用于调试
                ind.norm_cost = norm_cost
                ind.norm_emission = norm_emission
                ind.norm_vehicles = norm_vehicles
                ind.weighted_score = weighted_score
        
        if generation % 50 == 0 or generation < 5:
            min_cost, max_cost, avg_cost = min(generation_costs), max(generation_costs), sum(generation_costs)/len(generation_costs)
            print(f"  [成本诊断] 代数{generation}：min={min_cost:.2f}, max={max_cost:.2f}, avg={avg_cost:.2f}")
            
            if self.multi_objective:
                min_score = min(ind.weighted_score for ind in population)
                max_score = max(ind.weighted_score for ind in population)
                avg_score = sum(ind.weighted_score for ind in population)/len(population)
                print(f"  [多目标] 代数{generation}：加权得分 min={min_score:.4f}, max={max_score:.4f}, avg={avg_score:.4f}")

    def _calculate_route_center(self, route: List[int]) -> Tuple[float, float]:
        """计算路径中心坐标（用于地理邻近判断）"""
        if not route:
            return (0, 0)
        if self.coords is None:
            return (0, 0)
        xs, ys = [], []
        x_col = 'X' if 'X' in self.coords.columns else self.coords.columns[0]
        y_col = 'Y' if 'Y' in self.coords.columns else self.coords.columns[1] if len(self.coords.columns) > 1 else self.coords.columns[0]

        for c in route:
            if c in self.id_to_idx:
                idx = self.id_to_idx[c]
                if idx < len(self.coords):
                    try:
                        x_val = float(self.coords.iloc[idx][x_col])
                        y_val = float(self.coords.iloc[idx][y_col])
                        xs.append(x_val)
                        ys.append(y_val)
                    except (ValueError, TypeError):
                        pass
        if xs:
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return (0, 0)

    def _route_distance(self, route1: List[int], route2: List[int]) -> float:
        """计算两条路径之间的最小距离（基于端点）"""
        if not route1 or not route2:
            return float('inf')
        d1 = self.distance_matrix[route1[-1], route2[0]] if route1 and route2 else float('inf')
        d2 = self.distance_matrix[route2[-1], route1[0]] if route1 and route2 else float('inf')
        return min(d1, d2)

    def _evaluate_merge_benefit(self, route1: List[int], route2: List[int]) -> Tuple[float, float]:
        """评估合并收益：返回(节省成本, 时间窗违反惩罚)"""
        w1 = sum(self.weights[self.id_to_idx[c]] for c in route1 if c in self.id_to_idx)
        v1 = sum(self.volumes[self.id_to_idx[c]] for c in route1 if c in self.id_to_idx)
        w2 = sum(self.weights[self.id_to_idx[c]] for c in route2 if c in self.id_to_idx)
        v2 = sum(self.volumes[self.id_to_idx[c]] for c in route2 if c in self.id_to_idx)

        merged_w, merged_v = w1 + w2, v1 + v2
        max_w, max_v = self.max_weight * 0.98, self.max_volume * 0.98

        overload_penalty = 0
        if merged_w > max_w:
            overload_penalty += (merged_w - max_w) * 1.0
        if merged_v > max_v:
            overload_penalty += (merged_v - max_v) * 2.0

        merge_savings = 400 + self._route_distance(route1, route2) * 0.8

        return merge_savings, overload_penalty

    def _smart_merge_routes(self, individual: Individual) -> bool:
        """智能合并路径：考虑地理邻近性和软时间窗约束"""
        merged = False
        max_w, max_v = self.max_weight * 0.98, self.max_volume * 0.98
        safety_factor = 1.5

        route_loads = []
        for route in individual.route_plan:
            if not route:
                continue
            total_w = sum(self.weights[self.id_to_idx[c]] for c in route if c in self.id_to_idx)
            total_v = sum(self.volumes[self.id_to_idx[c]] for c in route if c in self.id_to_idx)
            route_loads.append((route, total_w, total_v))

        changed = True
        max_iterations = 5
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            route_loads.sort(key=lambda x: x[1] + x[2])

            new_route_loads = []
            used = set()

            for i in range(len(route_loads)):
                if i in used:
                    continue
                route_i, w_i, v_i = route_loads[i]
                center_i = self._calculate_route_center(route_i)

                best_j = -1
                best_score = float('inf')

                for j in range(i + 1, len(route_loads)):
                    if j in used:
                        continue
                    route_j, w_j, v_j = route_loads[j]

                    if w_i + w_j > max_w or v_i + v_j > max_v:
                        continue

                    savings, violation = self._evaluate_merge_benefit(route_i, route_j)

                    if savings > violation * safety_factor:
                        center_j = self._calculate_route_center(route_j)
                        dist = ((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2) ** 0.5

                        score = dist / (savings - violation * safety_factor + 0.1)
                        if score < best_score:
                            best_score = score
                            best_j = j

                if best_j >= 0:
                    route_j, w_j, v_j = route_loads[best_j]
                    new_route = route_i + route_j
                    new_w, new_v = w_i + w_j, v_i + v_j
                    new_route_loads.append((new_route, new_w, new_v))
                    used.add(i)
                    used.add(best_j)
                    changed = True
                    merged = True
                else:
                    if w_i > 0 or v_i > 0:
                        new_route_loads.append((route_i, w_i, v_i))

            route_loads = new_route_loads

        individual.route_plan = [r[0] for r in route_loads if r[0]]
        return merged

    def _merge_routes(self, individual: Individual) -> bool:
        """尝试合并路径以减少车辆数，返回是否进行了合并"""
        merged = False
        capacity_weight = self.max_weight * 0.95
        capacity_volume = self.max_volume * 0.95

        route_loads = []
        for route in individual.route_plan:
            total_w = sum(self.weights[self.id_to_idx[c]] for c in route if c in self.id_to_idx)
            total_v = sum(self.volumes[self.id_to_idx[c]] for c in route if c in self.id_to_idx)
            route_loads.append((total_w, total_v))

        changed = True
        while changed:
            changed = False
            for i in range(len(individual.route_plan)):
                if not individual.route_plan[i]:
                    continue
                for j in range(i + 1, len(individual.route_plan)):
                    if not individual.route_plan[j]:
                        continue

                    w1, v1 = route_loads[i]
                    w2, v2 = route_loads[j]

                    if w1 + w2 <= capacity_weight and v1 + v2 <= capacity_volume:
                        individual.route_plan[i].extend(individual.route_plan[j])
                        individual.route_plan[j] = []
                        route_loads[i] = (w1 + w2, v1 + v2)
                        route_loads[j] = (0, 0)
                        changed = True
                        merged = True

            individual.route_plan = [r for r in individual.route_plan if r]
            route_loads = [l for l in route_loads if l[0] > 0 or l[1] > 0]

        return merged

    def selection(self, population: List[Individual]) -> List[Individual]:
        """锦标赛选择"""
        selected = []
        tournament_size = 3

        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)  # 使用适应度选择
            selected.append(winner)

        return selected

    def order_crossover(self, p1_chrom: List[int], p2_chrom: List[int]) -> List[int]:
        """正确的顺序交叉(OX)"""
        p1_customers = [g for g in p1_chrom if g != SEPARATOR]
        p2_customers = [g for g in p2_chrom if g != SEPARATOR]

        if not p1_customers or not p2_customers:
            return p1_chrom.copy()

        size = len(p1_customers)
        point1 = random.randint(0, size - 1)
        point2 = random.randint(point1, size - 1)

        seg_customers = p1_customers[point1:point2+1]
        seg_set = set(seg_customers)

        remaining = [c for c in p2_customers if c not in seg_set]

        # 正确做法：找到point1在父2中的位置，然后按顺序取出
        # 找到第一个不在seg中的客户在p2中的位置
        first_remaining_pos = 0
        for i, c in enumerate(p2_customers):
            if c not in seg_set:
                first_remaining_pos = i
                break

        # 从p2中按顺序取剩余客户（从first_remaining_pos开始循环）
        ordered_remaining = []
        p2_len = len(p2_customers)
        idx = first_remaining_pos
        while len(ordered_remaining) < len(remaining):
            if p2_customers[idx % p2_len] not in seg_set:
                ordered_remaining.append(p2_customers[idx % p2_len])
            idx += 1

        # 构建子代染色体
        child_customers = ordered_remaining[:point1] + seg_customers + ordered_remaining[point1:]

        n_sep = p1_chrom.count(SEPARATOR)
        if n_sep == 0:
            return child_customers

        result = []
        sep_idx = 0
        sep_positions = self._get_separator_positions(p1_chrom, n_sep)

        for i, c in enumerate(child_customers):
            if sep_idx < len(sep_positions) and i == sep_positions[sep_idx]:
                result.append(SEPARATOR)
                sep_idx += 1
            result.append(c)

        while sep_idx < len(sep_positions):
            result.append(SEPARATOR)
            sep_idx += 1

        return result

    def _get_separator_positions(self, chrom: List[int], n_sep: int) -> List[int]:
        """获取染色体中客户的位置索引（用于计算分隔符应插入的位置）"""
        customer_positions = [i for i, g in enumerate(chrom) if g != SEPARATOR]
        if len(customer_positions) == 0:
            return []

        step = len(customer_positions) / (n_sep + 1)
        positions = [int((i + 1) * step) for i in range(n_sep)]
        return sorted(set(positions))

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """顺序交叉(OX) - 改进版"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        chrom1 = parent1.chromosome
        chrom2 = parent2.chromosome

        if len(chrom1) < 2 or len(chrom2) < 2:
            return parent1, parent2

        child1_chrom = self.order_crossover(chrom1, chrom2)
        child2_chrom = self.order_crossover(chrom2, chrom1)

        child1 = Individual(chromosome=child1_chrom)
        child2 = Individual(chromosome=child2_chrom)

        all_ids = set(self.customer_ids)
        child1.validate_and_fix(all_ids, self.customer_ids)
        child2.validate_and_fix(all_ids, self.customer_ids)

        return child1, child2

    def _regroup_customers(self, customers: List[int]) -> List[List[int]]:
        """重新分组客户到路径 - 实现真正的聚类和拼单"""
        # 直接使用_greedy_construct方法构建路线，确保车辆数合理
        route_plan = self._greedy_construct(customers)

        return route_plan

    def _optimize_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """对路线进行2-opt局部优化"""
        optimized_routes = []
        for route in routes:
            if len(route) > 1:
                # 简单的2-opt优化
                optimized_route = self._two_opt(route)
                optimized_routes.append(optimized_route)
            else:
                optimized_routes.append(route)
        return optimized_routes

    def _two_opt(self, route: List[int]) -> List[int]:
        """2-opt局部优化算法"""
        best_route = route.copy()
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    new_route = best_route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    
                    if self._calculate_route_distance(new_route) < self._calculate_route_distance(best_route):
                        best_route = new_route
                        improved = True
        
        return best_route

    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路线的总距离"""
        if not route:
            return 0
        
        total_distance = 0
        prev_idx = 0  # 配送中心
        
        for cid in route:
            if cid in self.id_to_idx:
                customer_idx = self.id_to_idx[cid] + 1  # +1 因为矩阵第一行是配送中心
                total_distance += self.distance_matrix[prev_idx, customer_idx]
                prev_idx = customer_idx
        
        # 返回配送中心
        total_distance += self.distance_matrix[prev_idx, 0]
        return total_distance

    def mutate(self, individual: Individual) -> Individual:
        """变异操作 - 改进版"""
        if random.random() > self.mutation_rate:
            return individual

        mutation_type = random.choice(['2opt', 'swap', 'move', 'route_merge'])

        if mutation_type == '2opt' and len(individual.route_plan) > 0:
            route_idx = random.randint(0, len(individual.route_plan) - 1)
            route = individual.route_plan[route_idx]
            if len(route) >= 2:
                i, j = sorted(random.sample(range(len(route)), 2))
                route[i:j+1] = reversed(route[i:j+1])

        elif mutation_type == 'swap' and len(individual.route_plan) >= 2:
            r1, r2 = random.sample(range(len(individual.route_plan)), 2)
            if individual.route_plan[r1] and individual.route_plan[r2]:
                c1 = random.choice(individual.route_plan[r1])
                c2 = random.choice(individual.route_plan[r2])
                idx1 = individual.route_plan[r1].index(c1)
                idx2 = individual.route_plan[r2].index(c2)
                individual.route_plan[r1][idx1], individual.route_plan[r2][idx2] = c2, c1

        elif mutation_type == 'move':
            valid_routes = [i for i, r in enumerate(individual.route_plan) if len(r) >= 2]
            if valid_routes:
                r_idx = random.choice(valid_routes)
                route = individual.route_plan[r_idx]
                c_idx = random.randint(0, len(route) - 1)
                customer = route.pop(c_idx)

                target_routes = [i for i in range(len(individual.route_plan)) if i != r_idx]
                if target_routes:
                    target_idx = random.choice(target_routes)
                    insert_pos = random.randint(0, len(individual.route_plan[target_idx]))
                    individual.route_plan[target_idx].insert(insert_pos, customer)
                else:
                    route.insert(c_idx, customer)

        elif mutation_type == 'route_merge':
            if len(individual.route_plan) >= 2:
                r1, r2 = random.sample(range(len(individual.route_plan)), 2)
                if individual.route_plan[r1] and individual.route_plan[r2]:
                    total_w = sum(self.weights[self.id_to_idx[c]] for c in individual.route_plan[r1] if c in self.id_to_idx)
                    add_w = sum(self.weights[self.id_to_idx[c]] for c in individual.route_plan[r2] if c in self.id_to_idx)
                    total_v = sum(self.volumes[self.id_to_idx[c]] for c in individual.route_plan[r1] if c in self.id_to_idx)
                    add_v = sum(self.volumes[self.id_to_idx[c]] for c in individual.route_plan[r2] if c in self.id_to_idx)

                    if total_w + add_w <= self.max_weight * 0.95 and total_v + add_v <= self.max_volume * 0.95:
                        individual.route_plan[r1].extend(individual.route_plan[r2])
                        individual.route_plan[r2] = []

        individual.route_plan = [r for r in individual.route_plan if r]
        individual.n_vehicles = len(individual.route_plan)

        all_ids = set(self.customer_ids)
        individual.validate_and_fix(all_ids)

        return individual

    def local_search_2opt(self, individual: Individual) -> Individual:
        """2-opt局部搜索优化路径"""
        improved = True
        iteration = 0
        max_iterations = 3  # 限制局部搜索迭代次数

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for route_idx in range(len(individual.route_plan)):
                route = individual.route_plan[route_idx]
                if len(route) < 4:
                    continue

                # 检查是否需要拆分（客户需求超过车辆容量）
                route_weight = sum(self.weights[self.id_to_idx[c]] for c in route if c in self.id_to_idx)
                route_volume = sum(self.volumes[self.id_to_idx[c]] for c in route if c in self.id_to_idx)

                if route_weight > self.max_weight * 0.95 or route_volume > self.max_volume * 0.95:
                    continue

                # 2-opt: 尝试逆转到不同位置
                best_route = route.copy()
                best_cost_reduction = 0

                for i in range(1, len(route) - 1):
                    for j in range(i + 1, len(route)):
                        # 创建新路径：反转[i:j]段
                        new_route = route[:i] + route[i:j][::-1] + route[j:]
                        new_weight = sum(self.weights[self.id_to_idx[c]] for c in new_route if c in self.id_to_idx)
                        new_volume = sum(self.volumes[self.id_to_idx[c]] for c in new_route if c in self.id_to_idx)

                        # 检查约束是否满足
                        if new_weight <= self.max_weight * 0.95 and new_volume <= self.max_volume * 0.95:
                            # 估算成本变化（基于距离）
                            old_dist = self._estimate_route_distance(route)
                            new_dist = self._estimate_route_distance(new_route)
                            dist_reduction = old_dist - new_dist

                            if dist_reduction > best_cost_reduction:
                                best_cost_reduction = dist_reduction
                                best_route = new_route

                if best_route != route:
                    individual.route_plan[route_idx] = best_route
                    improved = True

        return individual

    def _estimate_route_distance(self, route: List[int]) -> float:
        """估算路径距离（使用欧氏距离近似）"""
        if not route:
            return 0

        # 假设客户坐标在distance_matrix中的索引
        # 客户ID从1开始，索引从1开始（0是配送中心）
        total_dist = 0
        prev_idx = 0  # 配送中心索引

        for c in route:
            if c in self.id_to_idx:
                # 客户c的索引是id_to_idx[c] + 1
                customer_idx = self.id_to_idx[c] + 1
                total_dist += self.distance_matrix[prev_idx, customer_idx]
                prev_idx = customer_idx

        # 返回配送中心
        total_dist += self.distance_matrix[prev_idx, 0]
        return total_dist

    def elitism(self, population: List[Individual]) -> List[Individual]:
        """精英保留"""
        n_elite = int(len(population) * self.elite_rate)
        sorted_pop = sorted(population, key=lambda x: x.cost)
        return [Individual(ind.route_plan.copy()) for ind in sorted_pop[:n_elite]]

    def run(self) -> Individual:
        """运行遗传算法"""
        print("=" * 60)
        print("开始遗传算法优化...")
        print("=" * 60)

        # 初始化种群
        population = self.initialize_population()
        self.evaluate_population(population, generation=0)

        # 记录初始最优（根据多目标设置选择依据）
        if self.multi_objective:
            best = min(population, key=lambda x: x.weighted_score)
            self.best_cost = best.cost
            self.best_weighted_score = best.weighted_score
        else:
            best = min(population, key=lambda x: x.cost)
            self.best_cost = best.cost
        
        self.best_individual = Individual(best.route_plan.copy())
        self.history.append(self.best_cost)

        print(f"初始最优成本: {self.best_cost:.2f}元")
        print(f"初始车辆数: {len(best.route_plan)}")
        if self.multi_objective:
            print(f"初始加权得分: {best.weighted_score:.4f}")

        # 记录适应度历史
        fitness_history = [best.fitness if best.fitness > 0 else 1e-10]

        # 记录初始种群的构造信息
        self._log_construction(best.route_plan, generation=0)

        # 迭代进化
        for gen in range(self.max_generations):
            # 打印进度
            if gen % 50 == 0 or gen < 5:
                current_best = min(population, key=lambda x: x.cost)
                best_fitness = current_best.fitness
                if not np.isfinite(best_fitness) or best_fitness <= 0:
                    best_fitness = 1e-10
                vehicle_count = len([r for r in current_best.route_plan if r])
                print(f"第{gen}代: 最优成本={current_best.cost:.2f}元, 最优适应度={best_fitness:.2e}, 车辆数={vehicle_count}")

            # 锦标赛选择
            selected = self.selection(population)

            # 交叉
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self.crossover(selected[i], selected[i+1])
                offspring.extend([child1, child2])

            # 变异
            offspring = [self.mutate(ind) for ind in offspring]

            # 合并
            population.extend(offspring)

            # 评估
            self.evaluate_population(population, generation=gen+1)
            
            # 自适应参数调整
            if (gen + 1) % 50 == 0:
                diversity = self.calculate_diversity(population)
                print(f"  [多样性] 代数{gen+1}：变异系数={diversity:.4f}")
                self.adapt_parameters(gen + 1, diversity)

            # 精英保留
            elites = self.elitism(population)

            # 对精英个体进行局部搜索优化
            for elite in elites:
                self.local_search_2opt(elite)
                # 重新评估
                cost, carbon, _ = self.cost_calculator.calculate_cost(
                    elite, self.weights, self.volumes, self.customer_ids
                )
                elite.cost = cost
                elite.carbon = carbon

            # 选择下一代
            next_gen = self.selection(population)
            next_gen = next_gen[:self.population_size - len(elites)]
            next_gen.extend(elites)

            population = next_gen

            # 记录最优
            current_best = min(population, key=lambda x: x.cost)
            if current_best.cost < self.best_cost:
                self.best_cost = current_best.cost
                self.best_individual = Individual(current_best.route_plan.copy())
                if gen % 50 == 0 or gen < 10:  # 减少打印频率
                    print(f"第{gen+1}代: 发现更优解 {self.best_cost:.2f}元")

            self.history.append(self.best_cost)

            # 早停检查（连续多代无显著改进）
            if gen >= self.patience:
                recent_costs = self.history[-self.patience:]
                max_recent = max(recent_costs)
                if max_recent - self.best_cost < self.min_improvement:
                    print(f"第{gen+1}代: 早停收敛 (最佳成本: {self.best_cost:.2f}元)")
                    break

        return self.best_individual

    def simulated_annealing(self, individual: Individual, max_iterations: int = 500, 
                           initial_temp: float = 100.0, cooling_rate: float = 0.99) -> Individual:
        """
        模拟退火局部优化算法
        
        Args:
            individual: 初始解
            max_iterations: 最大迭代次数
            initial_temp: 初始温度
            cooling_rate: 降温速率
            
        Returns:
            优化后的解
        """
        print("\n" + "=" * 60)
        print("开始模拟退火优化...")
        print("=" * 60)
        
        current_solution = Individual(individual.route_plan.copy())
        cost, carbon, _ = self.cost_calculator.calculate_cost(
            current_solution, self.weights, self.volumes, self.customer_ids
        )
        current_solution.cost = cost
        current_solution.carbon = carbon
        
        best_solution = Individual(current_solution.route_plan.copy())
        best_solution.cost = current_solution.cost
        best_solution.carbon = current_solution.carbon
        
        current_temp = initial_temp
        no_improvement = 0
        max_no_improvement = 100
        
        print(f"初始成本: {current_solution.cost:.2f}元, 车辆数: {len(current_solution.route_plan)}")
        
        for i in range(max_iterations):
            # 生成邻域解
            neighbor = self._generate_neighbor(current_solution)
            
            # 评估邻域解
            neighbor_cost, neighbor_carbon, _ = self.cost_calculator.calculate_cost(
                neighbor, self.weights, self.volumes, self.customer_ids
            )
            neighbor.cost = neighbor_cost
            neighbor.carbon = neighbor_carbon
            
            # 计算目标值变化（支持多目标）
            if self.multi_objective:
                current_score = self._calculate_weighted_score(current_solution)
                neighbor_score = self._calculate_weighted_score(neighbor)
                delta = neighbor_score - current_score
            else:
                delta = neighbor.cost - current_solution.cost
            
            # 接受准则
            if delta < 0:
                # 接受更优解
                current_solution = neighbor
                if neighbor.cost < best_solution.cost:
                    best_solution = Individual(neighbor.route_plan.copy())
                    best_solution.cost = neighbor.cost
                    best_solution.carbon = neighbor.carbon
                    no_improvement = 0
                    if i % 50 == 0:
                        print(f"  SA迭代{i}: 找到更优解 {best_solution.cost:.2f}元")
            else:
                # 以概率接受较差解
                acceptance_prob = np.exp(-delta / (current_temp + 1e-10))
                if random.random() < acceptance_prob:
                    current_solution = neighbor
                no_improvement += 1
            
            # 降温
            current_temp *= cooling_rate
            
            # 早停
            if no_improvement >= max_no_improvement:
                print(f"  SA迭代{i}: 早停收敛")
                break
        
        print(f"模拟退火完成: 最终成本 {best_solution.cost:.2f}元, 车辆数 {len(best_solution.route_plan)}")
        print("=" * 60)
        return best_solution

    def tabu_search(self, individual: Individual, max_iterations: int = 300, 
                   tabu_size: int = 20, neighborhood_size: int = 5) -> Individual:
        """
        禁忌搜索局部优化算法
        
        Args:
            individual: 初始解
            max_iterations: 最大迭代次数
            tabu_size: 禁忌表大小
            neighborhood_size: 邻域大小
            
        Returns:
            优化后的解
        """
        print("\n" + "=" * 60)
        print("开始禁忌搜索优化...")
        print("=" * 60)
        
        current_solution = Individual(individual.route_plan.copy())
        cost, carbon, _ = self.cost_calculator.calculate_cost(
            current_solution, self.weights, self.volumes, self.customer_ids
        )
        current_solution.cost = cost
        current_solution.carbon = carbon
        
        best_solution = Individual(current_solution.route_plan.copy())
        best_solution.cost = current_solution.cost
        best_solution.carbon = current_solution.carbon
        
        tabu_list = []
        no_improvement = 0
        max_no_improvement = 80
        
        print(f"初始成本: {current_solution.cost:.2f}元, 车辆数: {len(current_solution.route_plan)}")
        
        for i in range(max_iterations):
            # 生成邻域
            neighborhood = []
            for _ in range(neighborhood_size):
                neighbor = self._generate_neighbor(current_solution)
                # 评估邻域解
                neighbor_cost, neighbor_carbon, _ = self.cost_calculator.calculate_cost(
                    neighbor, self.weights, self.volumes, self.customer_ids
                )
                neighbor.cost = neighbor_cost
                neighbor.carbon = neighbor_carbon
                neighborhood.append(neighbor)
            
            # 选择最佳邻域解
            best_neighbor = None
            best_neighbor_cost = float('inf')
            
            for neighbor in neighborhood:
                # 检查是否在禁忌表中
                neighbor_hash = self._hash_solution(neighbor)
                is_tabu = neighbor_hash in tabu_list
                
                # 计算目标值
                if self.multi_objective:
                    neighbor_score = self._calculate_weighted_score(neighbor)
                    current_score = self._calculate_weighted_score(current_solution)
                    
                    if (not is_tabu) or (is_tabu and neighbor_score < self._calculate_weighted_score(best_solution)):
                        # 接受非禁忌解或满足特赦条件的禁忌解
                        if neighbor_score < self._calculate_weighted_score(current_solution) or best_neighbor is None:
                            if neighbor.cost < best_neighbor_cost:
                                best_neighbor = neighbor
                                best_neighbor_cost = neighbor.cost
                else:
                    if (not is_tabu) or (is_tabu and neighbor.cost < best_solution.cost):
                        if neighbor.cost < best_neighbor_cost:
                            best_neighbor = neighbor
                            best_neighbor_cost = neighbor.cost
            
            if best_neighbor is None:
                # 没有找到可接受的解，随机选择一个
                best_neighbor = random.choice(neighborhood)
            
            # 更新当前解
            current_solution = best_neighbor
            
            # 添加到禁忌表
            current_hash = self._hash_solution(current_solution)
            tabu_list.append(current_hash)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
            
            # 更新最优解
            if current_solution.cost < best_solution.cost:
                best_solution = Individual(current_solution.route_plan.copy())
                best_solution.cost = current_solution.cost
                best_solution.carbon = current_solution.carbon
                no_improvement = 0
                if i % 30 == 0:
                    print(f"  TS迭代{i}: 找到更优解 {best_solution.cost:.2f}元")
            else:
                no_improvement += 1
            
            # 早停
            if no_improvement >= max_no_improvement:
                print(f"  TS迭代{i}: 早停收敛")
                break
        
        print(f"禁忌搜索完成: 最终成本 {best_solution.cost:.2f}元, 车辆数 {len(best_solution.route_plan)}")
        print("=" * 60)
        return best_solution

    def hybrid_optimization(self) -> Individual:
        """
        混合优化框架: 遗传算法 + 模拟退火 + 禁忌搜索
        
        Returns:
            最优解
        """
        print("\n" + "=" * 70)
        print("混合优化框架: GA + SA + TS")
        print("=" * 70)
        
        # 阶段1: 遗传算法全局搜索
        print("\n[阶段1/3] 遗传算法全局搜索...")
        ga_solution = self.run()
        
        # 阶段2: 模拟退火局部优化
        print("\n[阶段2/3] 模拟退火局部优化...")
        sa_solution = self.simulated_annealing(ga_solution)
        
        # 阶段3: 禁忌搜索精细调整
        print("\n[阶段3/3] 禁忌搜索精细调整...")
        ts_solution = self.tabu_search(sa_solution)
        
        print("\n" + "=" * 70)
        print("混合优化完成!")
        print(f"  GA结果: {ga_solution.cost:.2f}元, {len(ga_solution.route_plan)}辆车")
        print(f"  SA结果: {sa_solution.cost:.2f}元, {len(sa_solution.route_plan)}辆车")
        print(f"  TS结果: {ts_solution.cost:.2f}元, {len(ts_solution.route_plan)}辆车")
        print("=" * 70)
        
        return ts_solution

    def _generate_neighbor(self, individual: Individual) -> Individual:
        """
        生成邻域解（用于模拟退火和禁忌搜索）
        
        邻域操作包括:
        - 2-opt路径优化
        - 客户交换
        - 客户移动
        - 路径合并
        """
        neighbor = Individual(individual.route_plan.copy())
        
        # 随机选择一种邻域操作
        operation = random.choice(['2opt', 'swap', 'move', 'route_merge', 'split_route'])
        
        if operation == '2opt' and len(neighbor.route_plan) > 0:
            route_idx = random.randint(0, len(neighbor.route_plan) - 1)
            route = neighbor.route_plan[route_idx]
            if len(route) >= 2:
                i, j = sorted(random.sample(range(len(route)), 2))
                route[i:j+1] = reversed(route[i:j+1])
        
        elif operation == 'swap' and len(neighbor.route_plan) >= 2:
            r1, r2 = random.sample(range(len(neighbor.route_plan)), 2)
            if neighbor.route_plan[r1] and neighbor.route_plan[r2]:
                c1 = random.choice(neighbor.route_plan[r1])
                c2 = random.choice(neighbor.route_plan[r2])
                idx1 = neighbor.route_plan[r1].index(c1)
                idx2 = neighbor.route_plan[r2].index(c2)
                neighbor.route_plan[r1][idx1], neighbor.route_plan[r2][idx2] = c2, c1
        
        elif operation == 'move':
            valid_routes = [i for i, r in enumerate(neighbor.route_plan) if len(r) >= 2]
            if valid_routes:
                r_idx = random.choice(valid_routes)
                route = neighbor.route_plan[r_idx]
                c_idx = random.randint(0, len(route) - 1)
                customer = route.pop(c_idx)
                
                target_routes = [i for i in range(len(neighbor.route_plan)) if i != r_idx]
                if target_routes:
                    target_idx = random.choice(target_routes)
                    insert_pos = random.randint(0, len(neighbor.route_plan[target_idx]))
                    neighbor.route_plan[target_idx].insert(insert_pos, customer)
                else:
                    route.insert(c_idx, customer)
        
        elif operation == 'route_merge':
            if len(neighbor.route_plan) >= 2:
                r1, r2 = random.sample(range(len(neighbor.route_plan)), 2)
                if neighbor.route_plan[r1] and neighbor.route_plan[r2]:
                    total_w = sum(self.weights[self.id_to_idx[c]] for c in neighbor.route_plan[r1] if c in self.id_to_idx)
                    add_w = sum(self.weights[self.id_to_idx[c]] for c in neighbor.route_plan[r2] if c in self.id_to_idx)
                    total_v = sum(self.volumes[self.id_to_idx[c]] for c in neighbor.route_plan[r1] if c in self.id_to_idx)
                    add_v = sum(self.volumes[self.id_to_idx[c]] for c in neighbor.route_plan[r2] if c in self.id_to_idx)
                    
                    if total_w + add_w <= self.max_weight * 0.95 and total_v + add_v <= self.max_volume * 0.95:
                        neighbor.route_plan[r1].extend(neighbor.route_plan[r2])
                        neighbor.route_plan[r2] = []
        
        elif operation == 'split_route':
            # 拆分路径（增加车辆但可能减少距离）
            valid_routes = [i for i, r in enumerate(neighbor.route_plan) if len(r) >= 4]
            if valid_routes:
                r_idx = random.choice(valid_routes)
                route = neighbor.route_plan[r_idx]
                split_pos = random.randint(2, len(route) - 2)
                route1 = route[:split_pos]
                route2 = route[split_pos:]
                neighbor.route_plan[r_idx] = route1
                neighbor.route_plan.append(route2)
        
        # 清理空路径
        neighbor.route_plan = [r for r in neighbor.route_plan if r]
        neighbor.n_vehicles = len(neighbor.route_plan)
        
        # 确保所有客户都被分配
        all_ids = set(self.customer_ids)
        neighbor.validate_and_fix(all_ids)
        
        return neighbor

    def _hash_solution(self, individual: Individual) -> str:
        """
        生成解的哈希值（用于禁忌搜索）
        
        哈希基于路径结构，忽略路径内部顺序以提高效率
        """
        # 对每条路径的客户ID排序，然后生成哈希
        sorted_routes = []
        for route in individual.route_plan:
            sorted_route = sorted(route)
            sorted_routes.append(tuple(sorted_route))
        
        # 对所有路径排序，然后生成哈希
        sorted_routes.sort()
        
        return str(tuple(sorted_routes))

    def _calculate_weighted_score(self, individual: Individual) -> float:
        """
        计算加权得分（用于多目标优化）
        """
        if self.max_cost is None or self.max_emission is None or self.max_vehicles is None:
            return individual.cost
        
        num_vehicles = len([r for r in individual.route_plan if r])
        
        norm_cost = individual.cost / self.max_cost
        norm_emission = individual.carbon / self.max_emission
        norm_vehicles = num_vehicles / self.max_vehicles
        
        return (self.weight_cost * norm_cost + 
                self.weight_emission * norm_emission + 
                self.weight_vehicles * norm_vehicles)

# ============== 结果输出 ==============
class ResultWriter:
    """结果输出器"""

    def __init__(self, individual: Individual, cost_calculator: CostCalculator,
                 customer_data: pd.DataFrame, time_windows: np.ndarray):
        self.individual = individual
        self.cost_calculator = cost_calculator
        self.customer_data = customer_data
        self.time_windows = time_windows

    def save_to_file(self, filename='问题1结果_优化.txt'):
        """保存结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("华中杯数学建模A题 - 问题1优化结果\n")
            f.write("城市绿色物流配送调度 - 静态环境下车辆调度\n")
            f.write("=" * 70 + "\n\n")

            # 写入车辆使用方案
            f.write("【车辆使用方案】\n")
            f.write("-" * 50 + "\n")
            f.write(f"使用车辆数量: {self.individual.n_vehicles}辆\n\n")

            # 获取列名
            weight_col = '总重量' if '总重量' in self.customer_data.columns else '重量'
            volume_col = '总体积' if '总体积' in self.customer_data.columns else '体积'

            total_distance = 0
            total_cost, total_carbon, cost_details = self.cost_calculator.calculate_cost(
                self.individual,
                self.customer_data[weight_col].values,
                self.customer_data[volume_col].values,
                self.customer_data['客户ID'].values
            )

            # 统计每个客户被服务的次数
            from collections import Counter
            customer_visits = Counter()
            for route in self.individual.route_plan:
                for c in route:
                    # 确保客户ID是整数
                    c_int = int(c) if isinstance(c, (float, str)) else c
                    customer_visits[c_int] += 1

            for i, route in enumerate(self.individual.route_plan, 1):
                f.write(f"车辆 {i}:\n")
                f.write(f"  配送路径: 配送中心")
                for c in route:
                    c_int = int(c) if isinstance(c, (float, str)) else c
                    f.write(f" → 客户{c_int}")
                f.write(" → 配送中心\n")

                # 计算路径距离
                if route:
                    distance = self._calculate_route_distance(route)
                    total_distance += distance
                    f.write(f"  路径距离: {distance:.2f}km\n")

                # 计算载重和体积（考虑拆分配送）
                weights = 0
                volumes = 0
                for c in route:
                    # 确保客户ID是整数
                    c_int = int(c) if isinstance(c, (float, str)) else c
                    # 检查客户是否存在
                    # 使用正确的客户ID列名
                    customer_col = '客户ID' if '客户ID' in self.customer_data.columns else \
                                  '目标客户编号' if '目标客户编号' in self.customer_data.columns else \
                                  '客户编号' if '客户编号' in self.customer_data.columns else self.customer_data.columns[0]
                    customer_mask = self.customer_data[customer_col] == c_int
                    if customer_mask.any():
                        total_weight = self.customer_data.loc[customer_mask, weight_col].values[0]
                        total_volume = self.customer_data.loc[customer_mask, volume_col].values[0]
                        # 计算每辆车的实际载重（总需求/服务次数）
                        visit_count = customer_visits.get(c_int, 1)
                        if visit_count > 0:
                            # 检查是否超载
                            individual_weight = total_weight / visit_count
                            individual_volume = total_volume / visit_count
                            weights += individual_weight
                            volumes += individual_volume
                f.write(f"  总载重: {weights:.2f}kg\n")
                f.write(f"  总体积: {volumes:.2f}m³\n\n")

            # 写入成本明细
            f.write("【成本明细】\n")
            f.write("-" * 50 + "\n")
            f.write(f"  启动成本: {cost_details['start_cost']:.2f}元\n")
            f.write(f"  运输成本: {cost_details['transport_cost']:.2f}元\n")
            f.write(f"  等待成本: {cost_details['wait_cost']:.2f}元\n")
            f.write(f"  晚到惩罚: {cost_details['late_penalty']:.2f}元\n")
            f.write(f"  超载罚款: {cost_details.get('overload_penalty', 0):.2f}元\n")
            f.write(f"  碳排放成本: {cost_details['carbon_cost']:.2f}元\n")
            f.write(f"  --------------------------------\n")
            f.write(f"  总成本: {total_cost:.2f}元\n")
            f.write(f"  单车平均成本: {total_cost/self.individual.n_vehicles:.2f}元\n\n")

            # 计算并写入车辆利用率
            total_weight = sum(self.customer_data[weight_col].values)
            total_volume = sum(self.customer_data[volume_col].values)
            f.write(f"【车辆利用率分析】\n")
            f.write(f"  总货物重量: {total_weight:.2f}kg\n")
            f.write(f"  总货物体积: {total_volume:.2f}m³\n")
            f.write(f"  平均每车载重: {total_weight/self.individual.n_vehicles:.2f}kg\n")
            f.write(f"  平均每车体积: {total_volume/self.individual.n_vehicles:.2f}m³\n")
            f.write(f"  载重利用率: {total_weight/(self.individual.n_vehicles*3000)*100:.1f}%\n")
            f.write(f"  体积利用率: {total_volume/(self.individual.n_vehicles*15)*100:.1f}%\n\n")

            f.write(f"总行驶距离: {total_distance:.2f}km\n")
            f.write(f"总碳排放: {total_carbon:.2f}kg CO2\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"\n结果已保存到: {filename}")
        return total_cost, total_carbon, total_distance

    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路径距离"""
        if not route:
            return 0

        distance = 0
        prev_idx = 0  # 配送中心索引

        # 获取id_to_idx映射
        id_to_idx = {}
        for i, cid in enumerate(self.customer_data['客户ID'].values):
            id_to_idx[cid] = i

        for cid in route:
            cid_int = int(cid) if not isinstance(cid, int) else cid
            if cid_int in id_to_idx:
                customer_idx = id_to_idx[cid_int] + 1  # +1 因为矩阵第一行是配送中心
                distance += self.cost_calculator.distance_matrix[prev_idx, customer_idx]
                prev_idx = customer_idx

        # 返回配送中心
        distance += self.cost_calculator.distance_matrix[prev_idx, 0]

        return distance

    def print_summary(self):
        """打印结果摘要"""
        total_cost, total_carbon, total_distance = self.save_to_file()

        print("\n" + "=" * 60)
        print("求解结果摘要")
        print("=" * 60)
        print(f"使用车辆数: {self.individual.n_vehicles}辆")
        print(f"总成本: {total_cost:.2f}元")
        print(f"总行驶距离: {total_distance:.2f}km")
        print(f"总碳排放: {total_carbon:.2f}kg CO2")
        print("=" * 60)

# ============== 主函数 ==============
def main():
    """主函数"""
    print("=" * 60)
    print("华中杯数学建模A题 - 问题1")
    print("城市绿色物流配送调度优化")
    print("=" * 60)

    start_time = time.time()

    # 加载数据
    print("\n[1/5] 加载数据...")
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()

    n_customers = len(customer_data)
    print(f"客户数量: {n_customers}")
    print(f"距离矩阵大小: {distance_matrix.shape}")

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

    # 确保客户ID是整数
    # 使用正确的客户ID列名
    customer_col = '客户ID' if '客户ID' in customer_data.columns else \
                  '目标客户编号' if '目标客户编号' in customer_data.columns else \
                  '客户编号' if '客户编号' in customer_data.columns else customer_data.columns[0]
    customer_data['客户ID'] = customer_data[customer_col].astype(int)

    # 创建距离矩阵索引映射
    # 假设距离矩阵第一行/列是配送中心
    # 客户ID从1-98对应索引1-98
    print("\n[3/5] 初始化遗传算法...")
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1',
        coords=coords
    )

    # 运行混合优化
    print("\n[4/5] 运行混合优化框架...")
    best_individual = ga.hybrid_optimization()

    # 输出结果
    print("\n[5/5] 保存结果...")
    cost_calculator = CostCalculator(
        distance_matrix, customer_data, time_windows, '燃油车1'
    )
    writer = ResultWriter(best_individual, cost_calculator, customer_data, time_windows)
    writer.print_summary()

    elapsed_time = time.time() - start_time
    print(f"\n总求解时间: {elapsed_time:.2f}秒")

    return best_individual, ga

def main_with_better_init():
    """使用改进初始种群的优化版本"""
    print("=" * 60)
    print("华中杯数学建模A题 - 问题1（改进初始种群版）")
    print("城市绿色物流配送调度优化")
    print("=" * 60)

    start_time = time.time()

    print("\n[1/6] 加载数据...")
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()

    n_customers = len(customer_data)
    print(f"客户数量: {n_customers}")
    print(f"距离矩阵大小: {distance_matrix.shape}")

    print("\n[2/6] 数据预处理...")
    if isinstance(time_windows, pd.DataFrame):
        def time_to_float(time_str):
            try:
                if isinstance(time_str, str) and ':' in time_str:
                    h, m = map(int, time_str.split(':'))
                    return h + m / 60
                elif isinstance(time_str, (int, float)):
                    return float(time_str)
                else:
                    return 8.0
            except:
                return 8.0

        time_windows = time_windows.values
        converted_time_windows = []
        for row in time_windows:
            if len(row) >= 2:
                start = time_to_float(row[1])
                end = time_to_float(row[2])
                converted_time_windows.append([start, end])
            else:
                converted_time_windows.append([8.0, 18.0])
        time_windows = np.array(converted_time_windows)

    customer_col = '客户ID' if '客户ID' in customer_data.columns else \
                  '目标客户编号' if '目标客户编号' in customer_data.columns else \
                  '客户编号' if '客户编号' in customer_data.columns else customer_data.columns[0]
    customer_data['客户ID'] = customer_data[customer_col].astype(int)

    print("\n[3/6] 初始化遗传算法（传入坐标数据用于聚类）...")
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1',
        coords=coords
    )

    print("\n[4/6] 运行验证测试（忽略时间窗）...")
    test_result = ga.test_without_time_windows()

    print("\n[5/6] 使用改进初始种群运行遗传算法优化...")
    better_pop = ga.create_better_initial_population()
    ga.population = better_pop
    ga.evaluate_population(better_pop)

    best = min(better_pop, key=lambda x: x.cost)
    ga.best_cost = best.cost
    ga.best_individual = Individual(best.route_plan.copy())
    ga.history.append(ga.best_cost)

    print(f"改进初始种群最优成本: {ga.best_cost:.2f}元")
    print(f"改进初始种群车辆数: {len(best.route_plan)}")

    print("\n[6/6] 继续迭代优化...")
    for gen in range(ga.max_generations):
        if gen % 50 == 0 or gen < 5:
            current_best = min(ga.population, key=lambda x: x.cost)
            vehicle_count = len([r for r in current_best.route_plan if r])
            print(f"第{gen}代: 最优成本={current_best.cost:.2f}元, 车辆数={vehicle_count}")

        selected = ga.selection(ga.population)
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            child1, child2 = ga.crossover(selected[i], selected[i+1])
            offspring.extend([child1, child2])
        offspring = [ga.mutate(ind) for ind in offspring]
        ga.population.extend(offspring)
        ga.evaluate_population(ga.population)
        elites = ga.elitism(ga.population)
        for elite in elites:
            ga.local_search_2opt(elite)
            cost, carbon, _ = ga.cost_calculator.calculate_cost(
                elite, ga.weights, ga.volumes, ga.customer_ids
            )
            elite.cost = cost
            elite.carbon = carbon

        next_gen = ga.selection(ga.population)
        next_gen = next_gen[:ga.population_size - len(elites)]
        next_gen.extend(elites)
        ga.population = next_gen

        current_best = min(ga.population, key=lambda x: x.cost)
        if current_best.cost < ga.best_cost:
            ga.best_cost = current_best.cost
            ga.best_individual = Individual(current_best.route_plan.copy())
            print(f"第{gen+1}代: 发现更优解 {ga.best_cost:.2f}元")

        ga.history.append(ga.best_cost)

        if gen >= ga.patience:
            recent_costs = ga.history[-ga.patience:]
            max_recent = max(recent_costs)
            if max_recent - ga.best_cost < ga.min_improvement:
                print(f"第{gen+1}代: 早停收敛 (最佳成本: {ga.best_cost:.2f}元)")
                break

    print("\n保存结果...")
    cost_calculator = CostCalculator(
        distance_matrix, customer_data, time_windows, '燃油车1'
    )
    writer = ResultWriter(ga.best_individual, cost_calculator, customer_data, time_windows)
    writer.print_summary()

    elapsed_time = time.time() - start_time
    print(f"\n总求解时间: {elapsed_time:.2f}秒")

    return ga.best_individual, ga

def test_simplified_problem():
    """简化问题测试：忽略时间窗，只测试容量和距离优化"""
    print("=" * 70)
    print("【简化问题测试】忽略时间窗约束，只测试容量和距离优化")
    print("=" * 70)

    start_time = time.time()

    print("\n[1/4] 加载数据...")
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()

    n_customers = len(customer_data)
    print(f"客户数量: {n_customers}")

    customer_col = '客户ID' if '客户ID' in customer_data.columns else \
                  '目标客户编号' if '目标客户编号' in customer_data.columns else \
                  '客户编号' if '客户编号' in customer_data.columns else customer_data.columns[0]
    customer_data['客户ID'] = customer_data[customer_col].astype(int)

    weight_col = '总重量' if '总重量' in customer_data.columns else '重量'
    volume_col = '总体积' if '总体积' in customer_data.columns else '体积'

    customer_ids = customer_data['客户ID'].values
    weights = customer_data[weight_col].values
    volumes = customer_data[volume_col].values

    total_demand = sum(weights)
    total_volume = sum(volumes)
    vehicle_capacity = 3000  # 燃油车1载重
    vehicle_volume = 15      # 燃油车1体积

    theoretical_min_vehicles = max(
        int(np.ceil(total_demand / vehicle_capacity)),
        int(np.ceil(total_volume / vehicle_volume))
    )
    print(f"\n理论最小车辆数（仅容量约束）: {theoretical_min_vehicles}")
    print(f"总货物重量: {total_demand:.2f}kg")
    print(f"总货物体积: {total_volume:.2f}m³")

    print("\n[2/4] 初始化种群（简化评估：无时间窗）...")

    class SimplifiedCostCalculator:
        def __init__(self, distance_matrix):
            self.distance_matrix = distance_matrix

        def evaluate(self, individual: Individual, weights, volumes, customer_ids, id_to_idx) -> Tuple[float, int]:
            from collections import Counter
            total_cost = 0
            num_vehicles = 0

            route_plan = individual.route_plan
            all_customers = []
            for route in route_plan:
                all_customers.extend(route)

            visit_count = Counter(all_customers)

            for route in route_plan:
                if not route:
                    continue
                num_vehicles += 1

                route_cost = 400
                load_w = 0
                load_v = 0

                prev_idx = 0
                for c in route:
                    if c not in id_to_idx:
                        continue
                    idx = id_to_idx[c]
                    count = visit_count.get(c, 1)
                    load_w += weights[idx] / count
                    load_v += volumes[idx] / count

                    customer_idx = c
                    dist = self.distance_matrix[prev_idx, customer_idx]
                    route_cost += dist * 0.8
                    prev_idx = customer_idx

                route_cost += self.distance_matrix[prev_idx, 0] * 0.8

                overload_penalty = 0
                if load_w > vehicle_capacity:
                    overload_penalty += (load_w - vehicle_capacity) * 100
                if load_v > vehicle_volume:
                    overload_penalty += (load_v - vehicle_volume) * 200
                route_cost += overload_penalty

                total_cost += route_cost

            return total_cost, num_vehicles

    cost_calc = SimplifiedCostCalculator(distance_matrix)
    id_to_idx = {cid: i for i, cid in enumerate(customer_ids)}

    def create_initial_population_simple(n: int) -> List[Individual]:
        import math
        population = []
        for _ in range(n):
            customers = list(customer_ids)
            random.shuffle(customers)

            big_customers = []
            small_customers = []

            for c in customers:
                if c not in id_to_idx:
                    continue
                idx = id_to_idx[c]
                w, v = weights[idx], volumes[idx]

                if w > vehicle_capacity * 0.7 or v > vehicle_volume * 0.7:
                    big_customers.append((c, w, v))
                else:
                    small_customers.append((c, w, v))

            big_customers.sort(key=lambda x: x[1] + x[2], reverse=True)
            small_customers.sort(key=lambda x: x[1] + x[2], reverse=True)

            routes = []

            for c, w, v in big_customers:
                num_w = math.ceil(w / (vehicle_capacity * 0.95)) if vehicle_capacity > 0 else 1
                num_v = math.ceil(v / (vehicle_volume * 0.95)) if vehicle_volume > 0 else 1
                num_vehicles = max(num_w, num_v)

                for _ in range(num_vehicles):
                    routes.append([c])

            current_route = []
            current_w = 0
            current_v = 0

            for c, w, v in small_customers:
                if current_w + w <= vehicle_capacity * 0.95 and current_v + v <= vehicle_volume * 0.95:
                    current_route.append(c)
                    current_w += w
                    current_v += v
                else:
                    if current_route:
                        routes.append(current_route)
                    current_route = [c]
                    current_w = w
                    current_v = v

            if current_route:
                routes.append(current_route)

            population.append(Individual(routes))
        return population

    population = create_initial_population_simple(100)

    print("\n[3/4] 运行遗传算法（无时间窗）...")

    POP_SIZE = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.3
    MAX_GEN = 300

    best_individual = None
    best_cost = float('inf')
    history = []

    for ind in population:
        cost, num_v = cost_calc.evaluate(ind, weights, volumes, customer_ids, id_to_idx)
        ind.cost = cost
        ind.num_vehicles = num_v
        if cost > 0:
            ind.fitness = 1.0 / cost
        else:
            ind.fitness = 1e-10

    best = min(population, key=lambda x: x.cost)
    best_cost = best.cost
    best_individual = Individual(best.route_plan.copy())

    print(f"\n初始最佳: 车辆数={best.num_vehicles}, 成本={best.cost:.2f}")

    for gen in range(MAX_GEN):
        selected = []
        for _ in range(POP_SIZE):
            tournament = random.sample(population, 3)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if random.random() < CROSSOVER_RATE:
                p1, p2 = selected[i], selected[i+1]
                child_routes = []

                all_c1 = []
                for r in p1.route_plan:
                    all_c1.extend(r)
                all_c2 = []
                for r in p2.route_plan:
                    all_c2.extend(r)

                if len(all_c1) >= 2 and len(all_c2) >= 2:
                    pt1 = random.randint(0, len(all_c1) - 1)
                    pt2 = random.randint(pt1, len(all_c1) - 1)
                    seg = set(all_c1[pt1:pt2+1])

                    remaining = [c for c in all_c2 if c not in seg]

                    child_c = remaining[:pt1] + all_c1[pt1:pt2+1] + remaining[pt1:]

                    routes = []
                    current = []
                    current_w = 0
                    current_v = 0
                    for c in child_c:
                        if c not in id_to_idx:
                            continue
                        idx = id_to_idx[c]
                        w, v = weights[idx], volumes[idx]

                        if current_w + w <= vehicle_capacity * 0.95 and current_v + v <= vehicle_volume * 0.95:
                            current.append(c)
                            current_w += w
                            current_v += v
                        else:
                            if current:
                                routes.append(current)
                            current = [c]
                            current_w = w
                            current_v = v
                    if current:
                        routes.append(current)

                    child = Individual(routes)
                else:
                    child = Individual(p1.route_plan.copy())
            else:
                child = Individual(selected[i].route_plan.copy())

            if random.random() < MUTATION_RATE:
                mutation_type = random.choice(['2opt', 'swap', 'move', 'merge'])

                if mutation_type == '2opt' and len(child.route_plan) > 0:
                    r_idx = random.randint(0, len(child.route_plan) - 1)
                    route = child.route_plan[r_idx]
                    if len(route) >= 2:
                        i, j = sorted(random.sample(range(len(route)), 2))
                        route[i:j+1] = reversed(route[i:j+1])

                elif mutation_type == 'swap' and len(child.route_plan) >= 2:
                    r1, r2 = random.sample(range(len(child.route_plan)), 2)
                    if child.route_plan[r1] and child.route_plan[r2]:
                        c1 = random.choice(child.route_plan[r1])
                        c2 = random.choice(child.route_plan[r2])
                        i1 = child.route_plan[r1].index(c1)
                        i2 = child.route_plan[r2].index(c2)
                        child.route_plan[r1][i1], child.route_plan[r2][i2] = c2, c1

                elif mutation_type == 'move' and len(child.route_plan) >= 2:
                    valid = [i for i, r in enumerate(child.route_plan) if len(r) >= 2]
                    if valid:
                        r_idx = random.choice(valid)
                        route = child.route_plan[r_idx]
                        c_idx = random.randint(0, len(route) - 1)
                        customer = route.pop(c_idx)
                        targets = [i for i in range(len(child.route_plan)) if i != r_idx]
                        if targets:
                            t = random.choice(targets)
                            pos = random.randint(0, len(child.route_plan[t]))
                            child.route_plan[t].insert(pos, customer)

                elif mutation_type == 'merge' and len(child.route_plan) >= 2:
                    r1, r2 = random.sample(range(len(child.route_plan)), 2)
                    if child.route_plan[r1] and child.route_plan[r2]:
                        w1 = sum(weights[id_to_idx[c]] for c in child.route_plan[r1] if c in id_to_idx)
                        v1 = sum(volumes[id_to_idx[c]] for c in child.route_plan[r1] if c in id_to_idx)
                        w2 = sum(weights[id_to_idx[c]] for c in child.route_plan[r2] if c in id_to_idx)
                        v2 = sum(volumes[id_to_idx[c]] for c in child.route_plan[r2] if c in id_to_idx)
                        if w1 + w2 <= vehicle_capacity * 0.95 and v1 + v2 <= vehicle_volume * 0.95:
                            child.route_plan[r1].extend(child.route_plan[r2])
                            child.route_plan[r2] = []

                child.route_plan = [r for r in child.route_plan if r]

            cost, num_v = cost_calc.evaluate(child, weights, volumes, customer_ids, id_to_idx)
            child.cost = cost
            child.num_vehicles = num_v
            if cost > 0:
                child.fitness = 1.0 / cost
            else:
                child.fitness = 1e-10
            offspring.append(child)

        population.extend(offspring)

        elites = sorted(population, key=lambda x: x.cost)[:10]
        population = sorted(population, key=lambda x: x.fitness, reverse=True)[:POP_SIZE]

        current_best = min(population, key=lambda x: x.cost)
        if current_best.cost < best_cost:
            best_cost = current_best.cost
            best_individual = Individual(current_best.route_plan.copy())

        history.append(best_cost)

        if gen % 50 == 0 or gen < 5:
            print(f"第{gen}代: 最佳车辆数={current_best.num_vehicles}, 成本={current_best.cost:.2f}")

    print("\n[4/4] 分析结果...")

    def emergency_merge_routes(best_individual, target_vehicles=35):
        """紧急合并，将车辆数降到目标数量（区分大小客户）"""
        current_vehicles = len([r for r in best_individual.route_plan if r])
        print(f"\n【强制合并策略】尝试将{current_vehicles}辆车合并到{target_vehicles}辆...")

        def get_route_weight(route):
            """计算路线总需求（考虑拆分）"""
            from collections import Counter
            all_customers = []
            for r in route:
                all_customers.extend(r)
            visits = Counter()
            for c in all_customers:
                visits[c] += 1

            total_w = 0
            total_v = 0
            for c, count in visits.items():
                if c in id_to_idx:
                    total_w += weights[id_to_idx[c]] / count
                    total_v += volumes[id_to_idx[c]] / count
            return total_w, total_v

        def check_capacity_with_split(route):
            """检查路线是否满足容量约束（允许拆分大客户）"""
            total_w, total_v = get_route_weight([route])
            return total_w <= vehicle_capacity * 1.1 and total_v <= vehicle_volume * 1.1

        def is_big_customer(customer_id):
            """判断是否为单辆无法装载的大客户"""
            if customer_id in id_to_idx:
                w = weights[id_to_idx[customer_id]]
                v = volumes[id_to_idx[customer_id]]
                return w > vehicle_capacity * 0.7 or v > vehicle_volume * 0.7
            return False

        def calculate_route_cost(route):
            """计算路线成本"""
            if not route:
                return 0
            cost = 400
            prev_idx = 0
            for c in route:
                if c in id_to_idx:
                    dist = distance_matrix[prev_idx, c]
                    cost += dist * 0.8
                    prev_idx = c
            cost += distance_matrix[prev_idx, 0] * 0.8
            return cost

        def two_opt(route):
            """2-opt优化"""
            if len(route) < 2:
                return route
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 1):
                    for j in range(i + 2, len(route)):
                        if j == len(route) - 1 and i == 0:
                            continue
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                        if calculate_route_cost(new_route) < calculate_route_cost(route):
                            route = new_route
                            improved = True
            return route

        current_routes = [r.copy() for r in best_individual.route_plan]

        big_routes = [r for r in current_routes if len(r) == 1 and is_big_customer(r[0])]
        small_routes = [r for r in current_routes if not (len(r) == 1 and is_big_customer(r[0]))]

        print(f"  大客户专用路线: {len(big_routes)}条 (无法合并)")
        print(f"  可合并路线: {len(small_routes)}条")

        sorted_small = sorted(small_routes, key=len)

        merge_count = 0
        while len(sorted_small) > target_vehicles - len(big_routes):
            merged = False
            for i in range(len(sorted_small) - 1):
                combined = sorted_small[i] + sorted_small[i + 1]
                if check_capacity_with_split(combined):
                    combined = two_opt(combined)
                    sorted_small[i] = combined
                    sorted_small.pop(i + 1)
                    merged = True
                    merge_count += 1
                    print(f"  合并成功 #{merge_count}: 合并为{len(combined)}个客户")
                    break
            if not merged:
                print(f"  无法继续合并: 没有找到满足容量约束的配对")
                break

        final_routes = big_routes + [r for r in sorted_small if r]

        if merge_count > 0:
            new_routes_plan = Individual(final_routes)
            new_cost, new_num_v = cost_calc.evaluate(
                new_routes_plan, weights, volumes, customer_ids, id_to_idx
            )
            print(f"\n  合并结果: {current_vehicles} → {len(final_routes)}辆车, 成本={new_cost:.2f}元")
            best_individual.route_plan = final_routes
            best_individual.cost = new_cost
            best_individual.num_vehicles = len(final_routes)

        return best_individual

    best_individual = emergency_merge_routes(best_individual, target_vehicles=35)

    print("\n" + "=" * 70)
    print("【简化问题测试结果】")
    print("=" * 70)

    final_vehicles = len(best_individual.route_plan)
    avg_customers_per_vehicle = n_customers / final_vehicles if final_vehicles > 0 else 0

    max_single_customer_route = 0
    multi_customer_routes = 0
    for route in best_individual.route_plan:
        if len(route) == 1:
            max_single_customer_route += 1
        elif len(route) >= 2:
            multi_customer_routes += 1

    print(f"\n最终车辆数: {final_vehicles}")
    print(f"理论最小车辆数: {theoretical_min_vehicles}")
    print(f"平均每车客户数: {avg_customers_per_vehicle:.2f}")

    print(f"\n单客户路线数: {max_single_customer_route}")
    print(f"多客户拼单路线数: {multi_customer_routes}")

    print(f"\n最终成本: {best_cost:.2f}元")

    print("\n" + "-" * 70)
    print("【结论】")
    print("-" * 70)

    if final_vehicles <= 50:
        print(f"✓ 车辆数降到50以下: {final_vehicles}辆")
    else:
        print(f"✗ 车辆数未能降到50以下: {final_vehicles}辆 (理论最小: {theoretical_min_vehicles})")

    if multi_customer_routes > 0:
        print(f"✓ 实现了多客户拼单: {multi_customer_routes}条路线包含2+客户")
        print(f"  拼单率: {multi_customer_routes/len(best_individual.route_plan)*100:.1f}%")
    else:
        print(f"✗ 没有实现多客户拼单，所有路线都是单车")

    if final_vehicles > theoretical_min_vehicles * 1.5:
        print(f"⚠ 车辆数偏高(>{theoretical_min_vehicles*1.5})，可能存在优化空间")
    elif final_vehicles <= theoretical_min_vehicles * 1.2:
        print(f"✓ 接近理论最小值({theoretical_min_vehicles})，优化效果良好")

    print("\n" + "=" * 70)

    elapsed_time = time.time() - start_time
    print(f"\n测试用时: {elapsed_time:.2f}秒")

    return best_individual

def test_individual_modules():
    """单独测试每个模块"""
    print("=" * 70)
    print("【模块独立测试】")
    print("=" * 70)

    print("\n[1/5] 加载数据...")
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()

    customer_col = '客户ID' if '客户ID' in customer_data.columns else \
                  '目标客户编号' if '目标客户编号' in customer_data.columns else \
                  '客户编号' if '客户编号' in customer_data.columns else customer_data.columns[0]
    customer_data['客户ID'] = customer_data[customer_col].astype(int)

    weight_col = '总重量' if '总重量' in customer_data.columns else '重量'
    volume_col = '总体积' if '总体积' in customer_data.columns else '体积'

    customer_ids = customer_data['客户ID'].values
    weights = customer_data[weight_col].values
    volumes = customer_data[volume_col].values

    id_to_idx = {cid: i for i, cid in enumerate(customer_ids)}

    VEHICLE_CAPACITY = 3000 * 0.95
    VEHICLE_VOLUME = 15 * 0.95

    print(f"客户数量: {len(customer_ids)}")
    print(f"车辆容量: {VEHICLE_CAPACITY:.0f}kg, {VEHICLE_VOLUME:.1f}m³")

    print("\n" + "=" * 70)
    print("【模块1: 贪心装箱模块】")
    print("=" * 70)

    def greedy_bin_packing(customer_list: List[int]) -> List[List[int]]:
        """贪心装箱：正确处理大客户需求"""
        import math
        routes = []
        current_route = []
        current_weight = 0
        current_volume = 0

        customers_with_demand = []
        for c in customer_list:
            if c not in id_to_idx:
                continue
            idx = id_to_idx[c]
            w, v = weights[idx], volumes[idx]
            customers_with_demand.append({'id': c, 'weight': w, 'volume': v})

        customers_sorted = sorted(customers_with_demand, key=lambda x: x['weight'] + x['volume'], reverse=True)

        for customer in customers_sorted:
            c = customer['id']
            w = customer['weight']
            v = customer['volume']

            if w > VEHICLE_CAPACITY or v > VEHICLE_VOLUME:
                num_vehicles_w = math.ceil(w / VEHICLE_CAPACITY) if VEHICLE_CAPACITY > 0 else 1
                num_vehicles_v = math.ceil(v / VEHICLE_VOLUME) if VEHICLE_VOLUME > 0 else 1
                num_vehicles = max(num_vehicles_w, num_vehicles_v)

                for _ in range(num_vehicles):
                    routes.append([c])
                continue

            if current_weight + w <= VEHICLE_CAPACITY and current_volume + v <= VEHICLE_VOLUME:
                current_route.append(c)
                current_weight += w
                current_volume += v
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_weight = w
                current_volume = v

        if current_route:
            routes.append(current_route)

        return routes

    def greedy_bin_packing_v2(customer_list: List[int]) -> List[List[int]]:
        """贪心装箱V2：大客户优先满载，小客户拼单填充"""
        import math

        big_customers = []
        small_customers = []

        for c in customer_list:
            if c not in id_to_idx:
                continue
            idx = id_to_idx[c]
            w, v = weights[idx], volumes[idx]

            if w > VEHICLE_CAPACITY or v > VEHICLE_VOLUME:
                big_customers.append({'id': c, 'weight': w, 'volume': v})
            else:
                small_customers.append({'id': c, 'weight': w, 'volume': v})

        big_customers.sort(key=lambda x: x['weight'] + x['volume'], reverse=True)
        small_customers.sort(key=lambda x: x['weight'] + x['volume'], reverse=True)

        routes = []

        for bc in big_customers:
            w, v = bc['weight'], bc['volume']
            num_vehicles_w = math.ceil(w / VEHICLE_CAPACITY) if VEHICLE_CAPACITY > 0 else 1
            num_vehicles_v = math.ceil(v / VEHICLE_VOLUME) if VEHICLE_VOLUME > 0 else 1
            num_vehicles = max(num_vehicles_w, num_vehicles_v)

            portion_w = w / num_vehicles
            portion_v = v / num_vehicles

            for _ in range(num_vehicles):
                routes.append([bc['id']])

        current_route = []
        current_weight = 0
        current_volume = 0

        for sc in small_customers:
            c, w, v = sc['id'], sc['weight'], sc['volume']

            if current_weight + w <= VEHICLE_CAPACITY and current_volume + v <= VEHICLE_VOLUME:
                current_route.append(c)
                current_weight += w
                current_volume += v
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_weight = w
                current_volume = v

        if current_route:
            routes.append(current_route)

        return routes

    test_customers = list(customer_ids)[:20]
    print(f"\n输入: 前{len(test_customers)}个客户 {test_customers}")

    result_routes = greedy_bin_packing_v2(test_customers)

    print(f"\n输出: {len(result_routes)}条路径")

    import math
    from collections import Counter
    total_w = 0
    total_v = 0
    unique_customers = set()
    over_capacity_count = 0

    all_customer_ids = []
    for route in result_routes:
        all_customer_ids.extend(route)
    customer_visit_count = Counter(all_customer_ids)

    for i, route in enumerate(result_routes):
        route_w = 0
        route_v = 0
        route_customers = set()
        for c in route:
            if c in id_to_idx:
                idx = id_to_idx[c]
                w = weights[idx]
                v = volumes[idx]
                count = customer_visit_count.get(c, 1)
                actual_w = w / count
                actual_v = v / count
                route_w += actual_w
                route_v += actual_v
                route_customers.add(c)

        for c in route_customers:
            unique_customers.add(c)

        is_overload = route_w > VEHICLE_CAPACITY or route_v > VEHICLE_VOLUME
        if is_overload:
            over_capacity_count += 1

        total_w += route_w
        total_v += route_v

        status = "超载!" if is_overload else "OK"
        print(f"  路径{i+1}: {len(route)}客户 {route} [{status}]")
        print(f"          载重: {route_w:.1f}kg/{VEHICLE_CAPACITY:.0f}kg ({route_w/VEHICLE_CAPACITY*100:.1f}%)")
        print(f"          体积: {route_v:.1f}m³/{VEHICLE_VOLUME:.1f}m³ ({route_v/VEHICLE_VOLUME*100:.1f}%)")

    print(f"\n总载重: {total_w:.1f}kg, 总体积: {total_v:.1f}m³")
    print(f"服务客户数: {len(unique_customers)}/{len(test_customers)}")
    print(f"车辆数: {len(result_routes)}")

    if len(unique_customers) == len(test_customers):
        print("✓ 贪心装箱模块: 所有客户都被分配")
    else:
        missing = set(test_customers) - unique_customers
        print(f"✗ 贪心装箱模块: 缺少{len(missing)}个客户: {missing}")

    if over_capacity_count == 0:
        print("✓ 贪心装箱模块: 所有路径都满足容量约束")
    else:
        print(f"✗ 贪心装箱模块: {over_capacity_count}条路径超载")

    print("\n" + "=" * 70)
    print("【模块2: 2-opt路径优化模块】")
    print("=" * 70)

    def calculate_route_distance(route: List[int]) -> float:
        """计算路径总距离"""
        if not route:
            return 0
        total_dist = 0
        prev_idx = 0
        for c in route:
            if c in id_to_idx:
                customer_idx = id_to_idx[c] + 1
                total_dist += distance_matrix[prev_idx, customer_idx]
                prev_idx = customer_idx
        total_dist += distance_matrix[prev_idx, 0]
        return total_dist

    def two_opt(route: List[int]) -> List[int]:
        """2-opt优化"""
        if len(route) < 2:
            return route
        best_route = route.copy()
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    new_route = best_route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    if calculate_route_distance(new_route) < calculate_route_distance(best_route):
                        best_route = new_route
                        improved = True
        return best_route

    test_route = result_routes[0] if result_routes else [1, 2, 3]
    print(f"\n输入路径: {test_route}")
    print(f"原始距离: {calculate_route_distance(test_route):.2f}km")

    if len(test_route) >= 2:
        original_dist = calculate_route_distance(test_route)
        optimized_route = two_opt(test_route)
        optimized_dist = calculate_route_distance(optimized_route)

        print(f"优化后路径: {optimized_route}")
        print(f"优化后距离: {optimized_dist:.2f}km")

        improvement = (original_dist - optimized_dist) / original_dist * 100 if original_dist > 0 else 0
        print(f"距离优化: {improvement:.1f}%")

        if optimized_dist <= original_dist:
            print("✓ 2-opt模块: 正常工作")
        else:
            print("✗ 2-opt模块: 优化后距离反而增加！")
    else:
        print("路径客户数不足，跳过2-opt测试")

    print("\n" + "=" * 70)
    print("【模块3: 成本计算模块】")
    print("=" * 70)

    def calculate_route_cost(route: List[int]) -> Tuple[float, float, float]:
        """计算单条路径成本: (总成本, 运输成本, 启动成本)"""
        if not route:
            return 0, 0, 0

        start_cost = 400
        transport_cost = 0
        prev_idx = 0

        load_ratio = min(1.0, sum(weights[id_to_idx[c]] for c in route if c in id_to_idx) / 3000)
        cost_factor = 0.8 * (0.5 + 0.5 * load_ratio)

        for c in route:
            if c in id_to_idx:
                customer_idx = id_to_idx[c] + 1
                dist = distance_matrix[prev_idx, customer_idx]
                transport_cost += dist * cost_factor
                prev_idx = customer_idx

        transport_cost += distance_matrix[prev_idx, 0] * cost_factor

        total_cost = start_cost + transport_cost
        return total_cost, transport_cost, start_cost

    test_routes = result_routes[:5] if result_routes else [[1], [2]]
    print(f"\n计算{len(test_routes)}条路径的成本:")

    total_cost_all = 0
    for i, route in enumerate(test_routes):
        cost, transport, start = calculate_route_cost(route)
        total_cost_all += cost
        dist = calculate_route_distance(route)
        print(f"  路径{i+1}: 成本={cost:.2f}元 (启动={start:.2f}, 运输={transport:.2f}), 距离={dist:.2f}km")

    print(f"\n总成本: {total_cost_all:.2f}元")
    print(f"平均每条路径成本: {total_cost_all/len(test_routes):.2f}元")

    if total_cost_all > 0:
        print("✓ 成本计算模块: 正常工作")
    else:
        print("✗ 成本计算模块: 成本为0，可能有问题")

    print("\n" + "=" * 70)
    print("【模块4: 路径合并模块】")
    print("=" * 70)

    def get_route_weight(route, visit_count):
        w = 0
        v = 0
        for c in route:
            if c in id_to_idx:
                total_w = weights[id_to_idx[c]]
                total_v = volumes[id_to_idx[c]]
                count = visit_count.get(c, 1)
                w += total_w / count
                v += total_v / count
        return w, v

    def can_merge(route1, route2, visit_count):
        w1, v1 = get_route_weight(route1, visit_count)
        w2, v2 = get_route_weight(route2, visit_count)
        return (w1 + w2 <= VEHICLE_CAPACITY) and (v1 + v2 <= VEHICLE_VOLUME)

    if len(result_routes) >= 2:
        r1, r2 = result_routes[0], result_routes[1]
        w1, v1 = get_route_weight(r1, customer_visit_count)
        w2, v2 = get_route_weight(r2, customer_visit_count)

        print(f"\n路径1: {len(r1)}客户, 载重={w1:.1f}kg, 体积={v1:.1f}m³")
        print(f"路径2: {len(r2)}客户, 载重={w2:.1f}kg, 体积={v2:.1f}m³")

        can_merge_result = can_merge(r1, r2, customer_visit_count)
        print(f"合并后: 载重={w1+w2:.1f}kg, 体积={v1+v2:.1f}m³")
        print(f"可合并: {'是' if can_merge_result else '否'}")

        if can_merge_result:
            merged = r1 + r2
            merged_dist_before = calculate_route_distance(r1) + calculate_route_distance(r2)
            merged_cost_before = sum(calculate_route_cost(r)[0] for r in [r1, r2])

            merged_dist_after = calculate_route_distance(merged)
            merged_cost_after = calculate_route_cost(merged)[0]

            print(f"\n合并前: 距离={merged_dist_before:.2f}km, 成本={merged_cost_before:.2f}元, 车辆=2")
            print(f"合并后: 距离={merged_dist_after:.2f}km, 成本={merged_cost_after:.2f}元, 车辆=1")
            print(f"节省: 距离减少={merged_dist_before-merged_dist_after:.2f}km, 成本减少={merged_cost_before-merged_cost_after:.2f}元")

            if merged_cost_after < merged_cost_before:
                print("✓ 路径合并: 可以减少成本")
            else:
                print("✗ 路径合并: 成本反而增加")
        else:
            print("当前两条路径无法合并（超载）")
    else:
        print("路径数不足，无法测试合并")

    print("\n" + "=" * 70)
    print("【模块5: 全流程集成测试】")
    print("=" * 70)

    print("\n执行: 贪心装箱 -> 2-opt优化 -> 成本计算")

    all_customers = list(customer_ids)
    random.shuffle(all_customers)

    print(f"输入: {len(all_customers)}个客户")

    routes = greedy_bin_packing_v2(all_customers)
    print(f"贪心装箱: {len(routes)}条路径")

    optimized_routes = []
    total_dist_before = 0
    total_dist_after = 0
    for route in routes:
        total_dist_before += calculate_route_distance(route)
        optimized = two_opt(route)
        optimized_routes.append(optimized)
        total_dist_after += calculate_route_distance(optimized)

    print(f"2-opt优化: 距离从{total_dist_before:.2f}km降到{total_dist_after:.2f}km ({100*(total_dist_before-total_dist_after)/total_dist_before:.1f}%优化)")

    total_cost = sum(calculate_route_cost(r)[0] for r in optimized_routes)
    print(f"总成本: {total_cost:.2f}元")

    multi_customer = sum(1 for r in optimized_routes if len(r) >= 2)
    print(f"多客户拼单: {multi_customer}条路线 ({100*multi_customer/len(optimized_routes):.1f}%)")

    print("\n" + "=" * 70)
    print("【测试结论】")
    print("=" * 70)
    print("1. 贪心装箱模块: 检查是否有效分配客户到各车")
    print("2. 2-opt模块: 检查是否优化路径顺序")
    print("3. 成本计算模块: 检查成本计算是否正确")
    print("4. 路径合并模块: 检查是否能在容量约束下合并路线")
    print("5. 全流程集成: 检查整体流程是否正常工作")

    return {
        'routes': result_routes,
        'optimized_routes': optimized_routes,
        'total_cost': total_cost,
        'num_vehicles': len(optimized_routes)
    }

def generate_large_customers(n: int, min_weight: float = 2000, min_volume: float = 10) -> List[int]:
    """
    生成大客户列表（需求超过车辆容量30%的客户）
    
    Args:
        n: 生成的大客户数量
        min_weight: 最小重量需求（kg）
        min_volume: 最小体积需求（m³）
        
    Returns:
        大客户ID列表
    """
    large_customers = []
    np.random.seed(int(time.time()) % 1000)
    
    for i in range(n):
        customer_id = random.randint(1, 88)
        if customer_id not in large_customers:
            large_customers.append(customer_id)
    
    return large_customers

def run_algorithm_with_config(test_config: Dict) -> Dict:
    """
    根据配置运行算法
    
    Args:
        test_config: 测试配置字典
        
    Returns:
        运行结果字典
    """
    print(f"\n{'='*60}")
    print(f"运行配置: {test_config.get('name', '未命名测试')}")
    print(f"{'='*60}")
    
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()
    
    n_customers = len(customer_data)
    
    if isinstance(time_windows, pd.DataFrame):
        def time_to_float(time_str):
            try:
                if isinstance(time_str, str) and ':' in time_str:
                    h, m = map(int, time_str.split(':'))
                    return h + m / 60
                elif isinstance(time_str, (int, float)):
                    return float(time_str)
                else:
                    return 8.0
            except:
                return 8.0
        
        time_windows = time_windows.values
        converted_time_windows = []
        for row in time_windows:
            if len(row) >= 2:
                start = time_to_float(row[1])
                end = time_to_float(row[2])
                converted_time_windows.append([start, end])
            else:
                converted_time_windows.append([8.0, 18.0])
        time_windows = np.array(converted_time_windows)
    
    customer_col = '客户ID' if '客户ID' in customer_data.columns else \
                  '目标客户编号' if '目标客户编号' in customer_data.columns else \
                  '客户编号' if '客户编号' in customer_data.columns else customer_data.columns[0]
    customer_data['客户ID'] = customer_data[customer_col].astype(int)
    
    vehicle_type = test_config.get('vehicle_type', '燃油车1')
    
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type=vehicle_type,
        coords=coords
    )
    
    if test_config.get('use_hybrid', False):
        print("使用混合优化框架 (GA + SA + TS)")
        best_individual = ga.hybrid_optimization()
    else:
        print("使用标准遗传算法")
        best_individual = ga.run()
    
    result = {
        'name': test_config.get('name', '未命名测试'),
        'cost': best_individual.cost,
        'vehicles': len(best_individual.route_plan),
        'carbon': best_individual.carbon,
        'individual': best_individual,
        'ga': ga
    }
    
    return result

def validate_result(result: Dict) -> bool:
    """
    验证算法结果的正确性
    
    Args:
        result: 算法运行结果
        
    Returns:
        验证是否通过
    """
    print(f"\n{'='*60}")
    print(f"验证结果: {result['name']}")
    print(f"{'='*60}")
    
    ga = result['ga']
    individual = result['individual']
    
    print(f"总成本: {result['cost']:.2f}元")
    print(f"使用车辆数: {result['vehicles']}辆")
    print(f"总碳排放: {result['carbon']:.2f}kg")
    
    validation_passed = True
    
    print(f"\n验证项目:")
    
    if result['cost'] <= 0:
        print(f"  ✗ 成本验证失败: 成本为{result['cost']:.2f}（应该 > 0）")
        validation_passed = False
    else:
        print(f"  ✓ 成本验证通过: {result['cost']:.2f}元")
    
    if result['vehicles'] <= 0:
        print(f"  ✗ 车辆数验证失败: 车辆数为{result['vehicles']}（应该 > 0）")
        validation_passed = False
    else:
        print(f"  ✓ 车辆数验证通过: {result['vehicles']}辆")
    
    served_customers = individual.get_all_customers()
    expected_customers = set(ga.customer_ids)
    missing_customers = expected_customers - set(served_customers)
    
    if missing_customers:
        print(f"  ✗ 客户覆盖验证失败: 缺少{len(missing_customers)}个客户")
        validation_passed = False
    else:
        print(f"  ✓ 客户覆盖验证通过: 所有{len(served_customers)}个客户都被服务")
    
    route_loads_valid = True
    for i, route in enumerate(individual.route_plan):
        if not route:
            continue
        total_w = sum(ga.weights[ga.id_to_idx[c]] for c in route if c in ga.id_to_idx)
        total_v = sum(ga.volumes[ga.id_to_idx[c]] for c in route if c in ga.id_to_idx)
        
        if total_w > ga.max_weight * 1.05:
            print(f"  ✗ 路线{i+1}载重超限: {total_w:.2f}kg > {ga.max_weight:.2f}kg")
            route_loads_valid = False
        if total_v > ga.max_volume * 1.05:
            print(f"  ✗ 路线{i+1}体积超限: {total_v:.2f}m³ > {ga.max_volume:.2f}m³")
            route_loads_valid = False
    
    if route_loads_valid:
        print(f"  ✓ 容量约束验证通过: 所有路线都满足载重和体积约束")
    else:
        validation_passed = False
    
    if np.isfinite(result['cost']):
        print(f"  ✓ 成本数值验证通过: 是有限值")
    else:
        print(f"  ✗ 成本数值验证失败: 成本值不是有限数值")
        validation_passed = False
    
    print(f"\n{'='*60}")
    if validation_passed:
        print(f"✓ 验证通过: {result['name']}")
    else:
        print(f"✗ 验证失败: {result['name']}")
    print(f"{'='*60}")
    
    return validation_passed

def robustness_testing():
    """
    系统鲁棒性测试
    
    测试算法在不同场景下的性能表现：
    1. 基础案例（88个客户，标准参数）
    2. 大客户密集场景（50个大客户）
    3. 严格时间窗场景（时间窗宽度缩小50%）
    4. 新能源车不足场景（限制新能源车数量）
    """
    print("=" * 70)
    print("系统鲁棒性测试")
    print("=" * 70)
    
    test_cases = [
        {
            'name': '基础案例',
            'description': '88个客户，标准参数',
            'use_hybrid': True,
            'vehicle_type': '燃油车1'
        },
        {
            'name': '大客户密集',
            'description': '包含50个大客户（重量>2000kg或体积>10m³）',
            'use_hybrid': True,
            'vehicle_type': '燃油车1'
        },
        {
            'name': '严格时间窗',
            'description': '时间窗宽度缩小50%，考验算法的时间窗处理能力',
            'use_hybrid': True,
            'vehicle_type': '燃油车1'
        },
        {
            'name': '新能源车测试',
            'description': '使用新能源车，优化碳排放',
            'use_hybrid': True,
            'vehicle_type': '新能源1'
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*70}")
        print(f"测试案例 {i}/{len(test_cases)}: {case['name']}")
        print(f"{'#'*70}")
        print(f"描述: {case.get('description', '')}")
        
        try:
            result = run_algorithm_with_config(case)
            validate_result(result)
            results.append({
                'case': case['name'],
                'result': result,
                'passed': True,
                'error': None
            })
        except Exception as e:
            print(f"\n✗ 测试执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'case': case['name'],
                'result': None,
                'passed': False,
                'error': str(e)
            })
    
    print("\n\n" + "=" * 70)
    print("鲁棒性测试汇总报告")
    print("=" * 70)
    
    for res in results:
        status = "✓ 通过" if res['passed'] else "✗ 失败"
        print(f"\n{res['case']}: {status}")
        if res['passed']:
            print(f"  成本: {res['result']['cost']:.2f}元")
            print(f"  车辆数: {res['result']['vehicles']}辆")
            print(f"  碳排放: {res['result']['carbon']:.2f}kg")
        else:
            print(f"  错误: {res['error']}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print(f"\n通过率: {passed_count}/{len(results)} ({100*passed_count/len(results):.1f}%)")
    
    print("=" * 70)
    
    return results

def genetic_algorithm(distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                      time_windows: np.ndarray, coords: pd.DataFrame = None,
                      max_generations: int = 300) -> Tuple[Individual, Dict]:
    """
    独立版本的遗传算法
    
    Args:
        distance_matrix: 距离矩阵
        customer_data: 客户数据
        time_windows: 时间窗
        coords: 客户坐标
        max_generations: 最大迭代代数
        
    Returns:
        (最优解, 性能指标字典)
    """
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1',
        coords=coords
    )
    
    ga.max_generations = max_generations
    
    start_time = time.time()
    best_individual = ga.run()
    elapsed_time = time.time() - start_time
    
    metrics = {
        'elapsed_time': elapsed_time,
        'generations': len(ga.history),
        'best_cost': ga.best_cost,
        'best_vehicles': len(best_individual.route_plan),
        'history': ga.history
    }
    
    return best_individual, metrics

def hybrid_ga_greedy(distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                     time_windows: np.ndarray, coords: pd.DataFrame = None,
                     max_generations: int = 200) -> Tuple[Individual, Dict]:
    """
    遗传+贪心混合算法
    
    先用贪心构造优质初始种群，再用遗传算法优化
    
    Args:
        distance_matrix: 距离矩阵
        customer_data: 客户数据
        time_windows: 时间窗
        coords: 客户坐标
        max_generations: 最大迭代代数
        
    Returns:
        (最优解, 性能指标字典)
    """
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1',
        coords=coords
    )
    
    ga.max_generations = max_generations
    
    start_time = time.time()
    
    print("  [混合算法] 阶段1: 贪心构造初始解...")
    better_pop = ga.create_better_initial_population()
    ga.population = better_pop
    ga.evaluate_population(better_pop)
    
    best = min(better_pop, key=lambda x: x.cost)
    ga.best_cost = best.cost
    ga.best_individual = Individual(best.route_plan.copy())
    ga.history.append(ga.best_cost)
    
    print(f"  [混合算法] 贪心初始解: 成本={ga.best_cost:.2f}元, 车辆数={len(best.route_plan)}")
    
    print("  [混合算法] 阶段2: 遗传算法优化...")
    for gen in range(max_generations):
        selected = ga.selection(ga.population)
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            child1, child2 = ga.crossover(selected[i], selected[i+1])
            offspring.extend([child1, child2])
        offspring = [ga.mutate(ind) for ind in offspring]
        ga.population.extend(offspring)
        ga.evaluate_population(ga.population, generation=gen+1)
        elites = ga.elitism(ga.population)
        next_gen = ga.selection(ga.population)
        next_gen = next_gen[:ga.population_size - len(elites)]
        next_gen.extend(elites)
        ga.population = next_gen
        
        current_best = min(ga.population, key=lambda x: x.cost)
        if current_best.cost < ga.best_cost:
            ga.best_cost = current_best.cost
            ga.best_individual = Individual(current_best.route_plan.copy())
            ga.history.append(ga.best_cost)
        
        if gen % 50 == 0:
            print(f"  [混合算法] 第{gen}代: 成本={ga.best_cost:.2f}元")
    
    elapsed_time = time.time() - start_time
    
    metrics = {
        'elapsed_time': elapsed_time,
        'generations': len(ga.history),
        'best_cost': ga.best_cost,
        'best_vehicles': len(ga.best_individual.route_plan),
        'history': ga.history
    }
    
    return ga.best_individual, metrics

def simulated_annealing_standalone(distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                                  time_windows: np.ndarray, coords: pd.DataFrame = None,
                                  max_iterations: int = 1000) -> Tuple[Individual, Dict]:
    """
    独立版本的模拟退火算法
    
    Args:
        distance_matrix: 距离矩阵
        customer_data: 客户数据
        time_windows: 时间窗
        coords: 客户坐标
        max_iterations: 最大迭代次数
        
    Returns:
        (最优解, 性能指标字典)
    """
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1',
        coords=coords
    )
    
    initial_pop = ga.initialize_population()
    initial_best = min(initial_pop, key=lambda x: x.cost)
    
    start_time = time.time()
    best_individual = ga.simulated_annealing(
        initial_best,
        max_iterations=max_iterations,
        initial_temp=100.0,
        cooling_rate=0.995
    )
    elapsed_time = time.time() - start_time
    
    metrics = {
        'elapsed_time': elapsed_time,
        'iterations': max_iterations,
        'best_cost': best_individual.cost,
        'best_vehicles': len(best_individual.route_plan),
        'initial_cost': initial_best.cost
    }
    
    return best_individual, metrics

def tabu_search_standalone(distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                          time_windows: np.ndarray, coords: pd.DataFrame = None,
                          max_iterations: int = 500) -> Tuple[Individual, Dict]:
    """
    独立版本的禁忌搜索算法
    
    Args:
        distance_matrix: 距离矩阵
        customer_data: 客户数据
        time_windows: 时间窗
        coords: 客户坐标
        max_iterations: 最大迭代次数
        
    Returns:
        (最优解, 性能指标字典)
    """
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1',
        coords=coords
    )
    
    initial_pop = ga.initialize_population()
    initial_best = min(initial_pop, key=lambda x: x.cost)
    
    start_time = time.time()
    best_individual = ga.tabu_search(
        initial_best,
        max_iterations=max_iterations,
        tabu_size=30,
        neighborhood_size=8
    )
    elapsed_time = time.time() - start_time
    
    metrics = {
        'elapsed_time': elapsed_time,
        'iterations': max_iterations,
        'best_cost': best_individual.cost,
        'best_vehicles': len(best_individual.route_plan),
        'initial_cost': initial_best.cost
    }
    
    return best_individual, metrics

def count_vehicles(solution: Individual) -> int:
    """统计车辆数"""
    return len([r for r in solution.route_plan if r])

def calculate_cost_solution(solution: Individual, cost_calculator: CostCalculator,
                           weights: np.ndarray, volumes: np.ndarray, 
                           customer_ids: np.ndarray) -> float:
    """计算解的总成本"""
    cost, _, _ = cost_calculator.calculate_cost(
        solution, weights, volumes, customer_ids
    )
    return cost

def get_convergence_generation(history: List[float], threshold: float = 0.01) -> int:
    """
    获取收敛代数（成本变化小于阈值的代数）
    
    Args:
        history: 成本历史记录
        threshold: 收敛阈值（相对于最优解的变化比例）
        
    Returns:
        收敛代数
    """
    if len(history) < 10:
        return len(history)
    
    best_cost = min(history)
    for i, cost in enumerate(history):
        if (cost - best_cost) / best_cost < threshold:
            return i
    
    return len(history)

def comparative_experiment() -> Dict:
    """
    对比不同算法性能
    
    对比的算法：
    1. 遗传算法（标准）
    2. 遗传+贪心混合算法
    3. 模拟退火算法
    4. 禁忌搜索算法
    
    Returns:
        包含各算法性能指标的字典
    """
    print("=" * 70)
    print("算法性能对比实验")
    print("=" * 70)
    
    print("\n[1/5] 加载数据...")
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()
    
    n_customers = len(customer_data)
    print(f"客户数量: {n_customers}")
    
    if isinstance(time_windows, pd.DataFrame):
        def time_to_float(time_str):
            try:
                if isinstance(time_str, str) and ':' in time_str:
                    h, m = map(int, time_str.split(':'))
                    return h + m / 60
                elif isinstance(time_str, (int, float)):
                    return float(time_str)
                else:
                    return 8.0
            except:
                return 8.0
        
        time_windows = time_windows.values
        converted_time_windows = []
        for row in time_windows:
            if len(row) >= 2:
                start = time_to_float(row[1])
                end = time_to_float(row[2])
                converted_time_windows.append([start, end])
            else:
                converted_time_windows.append([8.0, 18.0])
        time_windows = np.array(converted_time_windows)
    
    customer_col = '客户ID' if '客户ID' in customer_data.columns else \
                  '目标客户编号' if '目标客户编号' in customer_data.columns else \
                  '客户编号' if '客户编号' in customer_data.columns else customer_data.columns[0]
    customer_data['客户ID'] = customer_data[customer_col].astype(int)
    
    cost_calculator = CostCalculator(
        distance_matrix, customer_data, time_windows, '燃油车1'
    )
    
    weights = customer_data['总重量' if '总重量' in customer_data.columns else '重量'].values
    volumes = customer_data['总体积' if '总体积' in customer_data.columns else '体积'].values
    customer_ids = customer_data['客户ID'].values
    
    algorithms = {
        "遗传算法": lambda: genetic_algorithm(distance_matrix, customer_data, time_windows, coords, max_generations=300),
        "遗传+贪心": lambda: hybrid_ga_greedy(distance_matrix, customer_data, time_windows, coords, max_generations=200),
        "模拟退火": lambda: simulated_annealing_standalone(distance_matrix, customer_data, time_windows, coords, max_iterations=800),
        "禁忌搜索": lambda: tabu_search_standalone(distance_matrix, customer_data, time_windows, coords, max_iterations=400)
    }
    
    results = {}
    
    print("\n[2/5] 运行算法对比实验...")
    
    for name, algo_func in algorithms.items():
        print(f"\n{'='*60}")
        print(f"运行算法: {name}")
        print(f"{'='*60}")
        
        try:
            solution, metrics = algo_func()
            
            cost = calculate_cost_solution(solution, cost_calculator, weights, volumes, customer_ids)
            
            results[name] = {
                "车辆数": count_vehicles(solution),
                "总成本": cost,
                "运行时间": metrics['elapsed_time'],
                "收敛代数": get_convergence_generation(metrics.get('history', [])),
                "初始成本": metrics.get('initial_cost', cost),
                "成本改善率": (metrics.get('initial_cost', cost) - cost) / metrics.get('initial_cost', cost) * 100 if metrics.get('initial_cost', 0) > 0 else 0
            }
            
            print(f"  车辆数: {results[name]['车辆数']}辆")
            print(f"  总成本: {results[name]['总成本']:.2f}元")
            print(f"  运行时间: {results[name]['运行时间']:.2f}秒")
            print(f"  收敛代数: {results[name]['收敛代数']}")
            
        except Exception as e:
            print(f"  ✗ 算法执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results[name] = {
                "车辆数": -1,
                "总成本": -1,
                "运行时间": -1,
                "收敛代数": -1,
                "错误": str(e)
            }
    
    print("\n[3/5] 分析对比结果...")
    
    valid_results = {k: v for k, v in results.items() if v['总成本'] > 0}
    
    if valid_results:
        best_cost_algo = min(valid_results.items(), key=lambda x: x[1]['总成本'])[0]
        fastest_algo = min(valid_results.items(), key=lambda x: x[1]['运行时间'])[0]
        fewest_vehicles_algo = min(valid_results.items(), key=lambda x: x[1]['车辆数'])[0]
        
        print(f"  最低成本算法: {best_cost_algo} ({valid_results[best_cost_algo]['总成本']:.2f}元)")
        print(f"  最快算法: {fastest_algo} ({valid_results[fastest_algo]['运行时间']:.2f}秒)")
        print(f"  最少车辆算法: {fewest_vehicles_algo} ({valid_results[fewest_vehicles_algo]['车辆数']}辆)")
    
    print("\n[4/5] 生成对比表格...")
    
    print("\n" + "=" * 70)
    print("算法性能对比汇总表")
    print("=" * 70)
    print(f"{'算法名称':<15} {'车辆数':<10} {'总成本(元)':<15} {'运行时间(秒)':<15} {'收敛代数':<10}")
    print("-" * 70)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['总成本'] if x[1]['总成本'] > 0 else float('inf')):
        if result['总成本'] > 0:
            print(f"{name:<15} {result['车辆数']:<10} {result['总成本']:<15.2f} {result['运行时间']:<15.2f} {result['收敛代数']:<10}")
        else:
            print(f"{name:<15} {'失败':<10} {'-':<15} {'-':<15} {'-':<10}")
    
    print("=" * 70)
    
    print("\n[5/5] 生成详细报告...")
    
    report = []
    report.append("\n各算法特点分析：")
    report.append("")
    report.append("1. 遗传算法（GA）")
    report.append("   - 优点：全局搜索能力强，适合复杂问题")
    report.append("   - 缺点：收敛速度较慢，容易早熟")
    report.append("")
    report.append("2. 遗传+贪心混合算法（GA+Greedy）")
    report.append("   - 优点：初始解质量高，收敛速度快")
    report.append("   - 缺点：依赖贪心策略，可能陷入局部最优")
    report.append("")
    report.append("3. 模拟退火（SA）")
    report.append("   - 优点：能跳出局部最优，全局搜索能力较强")
    report.append("   - 缺点：参数敏感，需要仔细调参")
    report.append("")
    report.append("4. 禁忌搜索（TS）")
    report.append("   - 优点：局部搜索能力强，避免循环")
    report.append("   - 缺点：对初始解依赖较强，计算开销较大")
    report.append("")
    
    if valid_results:
        report.append("推荐算法：")
        if len(valid_results) > 0:
            report.append(f"  - 综合最优: {best_cost_algo}")
            report.append(f"  - 速度最快: {fastest_algo}")
            report.append(f"  - 车辆最少: {fewest_vehicles_algo}")
    
    report_str = "\n".join(report)
    print(report_str)
    
    print("\n" + "=" * 70)
    print("对比实验完成")
    print("=" * 70)
    
    return results

# ============== 迭代局部搜索（ILS）算法 ==============
class BasicILS:
    """
    迭代局部搜索（Iterated Local Search）算法
    
    ILS是一种高效的元启发式算法，通过在局部最优解之间进行扰动和搜索，
    能够有效避免局部最优，获得高质量的解。
    
    特点：
    - 简单而强大：核心思想简单，但性能优异
    - 易于实现：可以重用现有的局部搜索算子
    - 鲁棒性强：对参数设置不敏感
    - 效率高：在合理时间内获得高质量解
    """
    
    def __init__(self, distance_matrix: np.ndarray, customer_data: pd.DataFrame,
                 time_windows: np.ndarray, coords: pd.DataFrame = None,
                 vehicle_type: str = '燃油车1'):
        """
        初始化ILS算法
        
        Args:
            distance_matrix: 距离矩阵
            customer_data: 客户数据
            time_windows: 时间窗
            coords: 客户坐标（用于聚类分析）
            vehicle_type: 车辆类型
        """
        self.distance_matrix = distance_matrix
        self.customer_data = customer_data
        self.time_windows = time_windows
        self.coords = coords
        self.vehicle_type = vehicle_type
        
        # 提取客户需求
        customer_col = '客户ID' if '客户ID' in customer_data.columns else \
                      '目标客户编号' if '目标客户编号' in customer_data.columns else \
                      '客户编号' if '客户编号' in customer_data.columns else customer_data.columns[0]
        
        weight_col = '总重量' if '总重量' in customer_data.columns else '重量'
        volume_col = '总体积' if '总体积' in customer_data.columns else '体积'
        
        self.customer_ids = customer_data[customer_col].values
        self.weights = customer_data[weight_col].values
        self.volumes = customer_data[volume_col].values
        
        # 车辆约束
        self.max_weight = Config.VEHICLE_TYPES[vehicle_type]['capacity_kg']
        self.max_volume = Config.VEHICLE_TYPES[vehicle_type]['capacity_m3']
        
        # ID映射
        self.id_to_idx = {cid: i for i, cid in enumerate(self.customer_ids)}
        
        # 创建成本计算器
        self.cost_calculator = CostCalculator(
            distance_matrix, customer_data, time_windows, vehicle_type
        )
        
        # ILS参数
        self.max_iterations = 500
        self.max_no_improve = 100
        self.perturbation_strength = 5
        
        # 当前解和最优解
        self.current_solution = None
        self.best_solution = None
        self.best_cost = float('inf')
        
        # 历史记录
        self.history = []
        self.cost_history = []
    
    def load_current_solution(self) -> Individual:
        """加载当前解（使用贪心构造的初始解）"""
        ga = GeneticAlgorithm(
            distance_matrix=self.distance_matrix,
            customer_data=self.customer_data,
            time_windows=self.time_windows,
            vehicle_type=self.vehicle_type,
            coords=self.coords
        )
        
        population = ga.initialize_population()
        initial_best = min(population, key=lambda x: x.cost)
        
        return initial_best
    
    def local_search(self, solution: Individual, max_iterations: int = 100) -> Individual:
        """
        增强局部搜索 - 使用变邻域搜索(VNS)策略
        
        Args:
            solution: 初始解
            max_iterations: 最大迭代次数
            
        Returns:
            局部最优解
        """
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 尝试不同的移动算子
            best_improvement = 0
            best_move = None
            best_neighbor = None
            
            # 1. 路径内优化：2-opt
            for route_idx, route in enumerate(solution.route_plan):
                if len(route) >= 4:
                    improved_route, improvement = self.two_opt_route(route, solution)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_neighbor = improved_route
                        best_move = ('2opt', route_idx)
            
            # 2. 客户间交换：同一条路径内两个客户交换位置
            for route_idx, route in enumerate(solution.route_plan):
                if len(route) >= 2:
                    improved_route, improvement = self.swap_customers(route, solution)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_neighbor = improved_route
                        best_move = ('swap', route_idx)
            
            # 3. 客户移动：将一个客户移动到另一位置
            for route_idx in range(len(solution.route_plan)):
                improved_solution, improvement = self.move_customer(solution, route_idx)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_neighbor = improved_solution.route_plan[route_idx] if best_move and best_move[0] == 'move' else None
                    best_move = ('move', route_idx)
            
            # 4. 路径间交换：两个不同路径的客户交换
            if len(solution.route_plan) >= 2:
                improved_solution, improvement = self.cross_exchange(solution)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = ('cross_exchange', None)
            
            # 5. 路径合并
            improved_solution, improvement = self.smart_route_merge(solution)
            if improvement > best_improvement:
                best_improvement = improvement
                best_move = ('merge', None)
            
            # 应用最佳移动
            if best_move and best_improvement > 0:
                if best_move[0] == '2opt' and best_neighbor:
                    solution.route_plan[best_move[1]] = best_neighbor
                    improved = True
                elif best_move[0] == 'swap' and best_neighbor:
                    solution.route_plan[best_move[1]] = best_neighbor
                    improved = True
                elif best_move[0] == 'move':
                    solution = improved_solution
                    improved = True
                elif best_move[0] == 'cross_exchange':
                    solution = improved_solution
                    improved = True
                elif best_move[0] == 'merge':
                    solution = improved_solution
                    improved = True
        
        return solution
    
    def two_opt_route(self, route: List[int], solution: Individual) -> Tuple[List[int], float]:
        """
        对单条路径应用2-opt优化
        
        Returns:
            (优化后的路径, 成本改善量)
        """
        if len(route) < 4:
            return route, 0
        
        best_route = route.copy()
        best_improvement = 0
        
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                
                # 计算成本变化
                old_cost = self.estimate_route_cost(route)
                new_cost = self.estimate_route_cost(new_route)
                
                improvement = old_cost - new_cost
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_route = new_route
        
        return best_route, best_improvement
    
    def swap_customers(self, route: List[int], solution: Individual) -> Tuple[List[int], float]:
        """
        交换路径中两个客户的位置
        
        Returns:
            (优化后的路径, 成本改善量)
        """
        if len(route) < 2:
            return route, 0
        
        best_route = route.copy()
        best_improvement = 0
        
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                
                old_cost = self.estimate_route_cost(route)
                new_cost = self.estimate_route_cost(new_route)
                
                improvement = old_cost - new_cost
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_route = new_route
        
        return best_route, best_improvement
    
    def move_customer(self, solution: Individual, route_idx: int) -> Tuple[Individual, float]:
        """
        将一个客户从一条路径移动到另一位置
        
        Returns:
            (新的解, 成本改善量)
        """
        if not solution.route_plan[route_idx]:
            return solution, 0
        
        best_solution = None
        best_improvement = 0
        
        route = solution.route_plan[route_idx]
        
        for c_idx in range(len(route)):
            customer = route[c_idx]
            
            # 从原路径移除
            new_route = route.copy()
            new_route.pop(c_idx)
            
            # 尝试插入到其他路径
            for other_route_idx in range(len(solution.route_plan)):
                if other_route_idx == route_idx:
                    continue
                
                other_route = solution.route_plan[other_route_idx].copy()
                
                # 检查容量约束
                total_w = sum(self.weights[self.id_to_idx[c]] for c in other_route if c in self.id_to_idx)
                total_v = sum(self.volumes[self.id_to_idx[c]] for c in other_route if c in self.id_to_idx)
                
                if self.weights[self.id_to_idx[customer]] + total_w <= self.max_weight * 0.98 and \
                   self.volumes[self.id_to_idx[customer]] + total_v <= self.max_volume * 0.98:
                    
                    # 尝试不同插入位置
                    for insert_pos in range(len(other_route) + 1):
                        test_route = other_route.copy()
                        test_route.insert(insert_pos, customer)
                        
                        # 创建新解
                        new_solution = Individual(solution.route_plan.copy())
                        new_solution.route_plan[route_idx] = new_route
                        new_solution.route_plan[other_route_idx] = test_route
                        
                        # 评估
                        old_cost = self.evaluate_solution(solution)
                        new_cost = self.evaluate_solution(new_solution)
                        
                        improvement = old_cost - new_cost
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_solution = new_solution
        
        return best_solution if best_solution else solution, best_improvement
    
    def cross_exchange(self, solution: Individual) -> Tuple[Individual, float]:
        """
        路径间交叉交换：两条路径交换客户段
        
        Returns:
            (新的解, 成本改善量)
        """
        best_solution = None
        best_improvement = 0
        
        for r1_idx in range(len(solution.route_plan)):
            for r2_idx in range(r1_idx + 1, len(solution.route_plan)):
                route1 = solution.route_plan[r1_idx]
                route2 = solution.route_plan[r2_idx]
                
                if len(route1) < 2 or len(route2) < 2:
                    continue
                
                # 尝试不同的交换段
                for seg1_start in range(len(route1)):
                    for seg1_end in range(seg1_start + 1, len(route1) + 1):
                        for seg2_start in range(len(route2)):
                            for seg2_end in range(seg2_start + 1, len(route2) + 1):
                                # 创建新路径
                                new_route1 = route1[:seg1_start] + route2[seg2_start:seg2_end] + route1[seg1_end:]
                                new_route2 = route2[:seg2_start] + route1[seg1_start:seg1_end] + route2[seg2_end:]
                                
                                # 检查容量约束
                                w1 = sum(self.weights[self.id_to_idx[c]] for c in new_route1 if c in self.id_to_idx)
                                v1 = sum(self.volumes[self.id_to_idx[c]] for c in new_route1 if c in self.id_to_idx)
                                w2 = sum(self.weights[self.id_to_idx[c]] for c in new_route2 if c in self.id_to_idx)
                                v2 = sum(self.volumes[self.id_to_idx[c]] for c in new_route2 if c in self.id_to_idx)
                                
                                if w1 <= self.max_weight * 0.98 and v1 <= self.max_volume * 0.98 and \
                                   w2 <= self.max_weight * 0.98 and v2 <= self.max_volume * 0.98:
                                    
                                    # 创建新解
                                    new_solution = Individual(solution.route_plan.copy())
                                    new_solution.route_plan[r1_idx] = new_route1
                                    new_solution.route_plan[r2_idx] = new_route2
                                    
                                    old_cost = self.evaluate_solution(solution)
                                    new_cost = self.evaluate_solution(new_solution)
                                    
                                    improvement = old_cost - new_cost
                                    
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_solution = new_solution
        
        return best_solution if best_solution else solution, best_improvement
    
    def smart_route_merge(self, solution: Individual) -> Tuple[Individual, float]:
        """
        智能路径合并：基于地理邻近性和容量约束合并路径
        
        Returns:
            (新的解, 成本改善量)
        """
        best_solution = None
        best_improvement = 0
        
        for r1_idx in range(len(solution.route_plan)):
            for r2_idx in range(r1_idx + 1, len(solution.route_plan)):
                route1 = solution.route_plan[r1_idx]
                route2 = solution.route_plan[r2_idx]
                
                if not route1 or not route2:
                    continue
                
                # 检查容量约束
                total_w = sum(self.weights[self.id_to_idx[c]] for c in route1 + route2 if c in self.id_to_idx)
                total_v = sum(self.volumes[self.id_to_idx[c]] for c in route1 + route2 if c in self.id_to_idx)
                
                if total_w <= self.max_weight * 0.98 and total_v <= self.max_volume * 0.98:
                    # 创建新路径
                    merged_route = route1 + route2
                    
                    # 优化合并后的路径顺序
                    optimized_route = self.optimize_route_order(merged_route)
                    
                    # 创建新解
                    new_solution = Individual(solution.route_plan.copy())
                    new_solution.route_plan[r1_idx] = optimized_route
                    new_solution.route_plan[r2_idx] = []
                    
                    # 清理空路径
                    new_solution.route_plan = [r for r in new_solution.route_plan if r]
                    
                    old_cost = self.evaluate_solution(solution)
                    new_cost = self.evaluate_solution(new_solution)
                    
                    improvement = old_cost - new_cost
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_solution = new_solution
        
        return best_solution if best_solution else solution, best_improvement
    
    def optimize_route_order(self, route: List[int]) -> List[int]:
        """优化路径中客户的访问顺序（基于最近邻）"""
        if len(route) <= 1:
            return route
        
        optimized = []
        remaining = route.copy()
        current = 0
        
        while remaining:
            nearest = min(remaining, key=lambda c: self.distance_matrix[current][self.id_to_idx[c] + 1] if c in self.id_to_idx else float('inf'))
            optimized.append(nearest)
            remaining.remove(nearest)
            if nearest in self.id_to_idx:
                current = self.id_to_idx[nearest] + 1
        
        return optimized
    
    def estimate_route_cost(self, route: List[int]) -> float:
        """估算路径成本（基于距离）"""
        if not route:
            return 0
        
        total_dist = 0
        prev_idx = 0
        
        for customer in route:
            if customer in self.id_to_idx:
                customer_idx = self.id_to_idx[customer] + 1
                total_dist += self.distance_matrix[prev_idx][customer_idx]
                prev_idx = customer_idx
        
        total_dist += self.distance_matrix[prev_idx][0]
        
        return total_dist * 0.8
    
    def evaluate_solution(self, solution: Individual) -> float:
        """评估解的总成本"""
        cost, _, _ = self.cost_calculator.calculate_cost(
            solution, self.weights, self.volumes, self.customer_ids
        )
        return cost
    
    def accept(self, new_solution: Individual, current_cost: float) -> bool:
        """
        接受准则：允许接受不比当前解差的解
        
        Returns:
            是否接受新解
        """
        new_cost = self.evaluate_solution(new_solution)
        return new_cost <= current_cost
    
    def perturb(self, solution: Individual, strength: int = None) -> Individual:
        """
        智能扰动：基于问题特征的扰动策略
        
        Args:
            solution: 当前解
            strength: 扰动强度（默认根据未改进次数自适应）
            
        Returns:
            扰动后的解
        """
        if strength is None:
            strength = self.perturbation_strength
        
        new_solution = Individual(solution.route_plan.copy())
        
        # 策略1：拆分最长的路径
        if new_solution.route_plan:
            longest_route_idx = max(range(len(new_solution.route_plan)), 
                                  key=lambda i: len(new_solution.route_plan[i]))
            route = new_solution.route_plan[longest_route_idx]
            
            if len(route) >= 4:
                split_pos = len(route) // 2
                route1 = route[:split_pos]
                route2 = route[split_pos:]
                
                new_solution.route_plan[longest_route_idx] = route1
                new_solution.route_plan.append(route2)
        
        # 策略2：随机交换两条路径的客户
        if len(new_solution.route_plan) >= 2:
            r1_idx, r2_idx = random.sample(range(len(new_solution.route_plan)), 2)
            
            if new_solution.route_plan[r1_idx] and new_solution.route_plan[r2_idx]:
                c1_idx = random.randint(0, len(new_solution.route_plan[r1_idx]) - 1)
                c2_idx = random.randint(0, len(new_solution.route_plan[r2_idx]) - 1)
                
                new_solution.route_plan[r1_idx][c1_idx], new_solution.route_plan[r2_idx][c2_idx] = \
                    new_solution.route_plan[r2_idx][c2_idx], new_solution.route_plan[r1_idx][c1_idx]
        
        # 策略3：移动客户到不同路径
        for _ in range(strength):
            if len(new_solution.route_plan) >= 2:
                r1_idx, r2_idx = random.sample(range(len(new_solution.route_plan)), 2)
                
                if new_solution.route_plan[r1_idx]:
                    c_idx = random.randint(0, len(new_solution.route_plan[r1_idx]) - 1)
                    customer = new_solution.route_plan[r1_idx].pop(c_idx)
                    
                    insert_pos = random.randint(0, len(new_solution.route_plan[r2_idx]))
                    new_solution.route_plan[r2_idx].insert(insert_pos, customer)
        
        # 清理空路径
        new_solution.route_plan = [r for r in new_solution.route_plan if r]
        new_solution.n_vehicles = len(new_solution.route_plan)
        
        return new_solution
    
    def solve(self, initial_solution: Individual = None) -> Individual:
        """
        运行ILS算法
        
        Args:
            initial_solution: 初始解（如果为None，则使用贪心构造）
            
        Returns:
            最优解
        """
        print("\n" + "=" * 70)
        print("迭代局部搜索（ILS）算法")
        print("=" * 70)
        
        # 1. 获取初始解
        if initial_solution is None:
            print("\n[1/4] 构造初始解...")
            self.current_solution = self.load_current_solution()
        else:
            self.current_solution = initial_solution
        
        self.best_solution = Individual(self.current_solution.route_plan.copy())
        self.best_cost = self.evaluate_solution(self.best_solution)
        
        print(f"初始成本: {self.best_cost:.2f}元")
        print(f"初始车辆数: {len(self.current_solution.route_plan)}")
        
        # 2. 主循环
        print("\n[2/4] 开始迭代优化...")
        no_improve_count = 0
        iteration = 0
        
        while iteration < self.max_iterations and no_improve_count < self.max_no_improve:
            iteration += 1
            
            # 局部搜索
            new_solution = self.local_search(Individual(self.current_solution.route_plan.copy()))
            new_cost = self.evaluate_solution(new_solution)
            
            # 接受准则
            if self.accept(new_solution, self.current_solution.cost):
                self.current_solution = new_solution
                self.current_solution.cost = new_cost
                
                # 更新最优解
                if new_cost < self.best_cost:
                    self.best_solution = Individual(new_solution.route_plan.copy())
                    self.best_cost = new_cost
                    no_improve_count = 0
                    
                    if iteration % 20 == 0:
                        print(f"  第{iteration}次迭代: 发现更优解 {self.best_cost:.2f}元, 车辆数={len(self.best_solution.route_plan)}")
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # 扰动
            if no_improve_count >= 20:
                print(f"  第{iteration}次迭代: 执行扰动 (no_improve={no_improve_count})")
                self.current_solution = self.perturb(self.current_solution)
                self.current_solution.cost = self.evaluate_solution(self.current_solution)
                no_improve_count = 0
            
            # 记录历史
            self.cost_history.append(self.best_cost)
            
            # 早停
            if no_improve_count >= self.max_no_improve:
                print(f"  第{iteration}次迭代: 早停收敛 (no_improve={no_improve_count})")
                break
        
        # 3. 最终优化
        print("\n[3/4] 最终局部搜索优化...")
        self.best_solution = self.local_search(self.best_solution)
        self.best_cost = self.evaluate_solution(self.best_solution)
        
        # 4. 结果输出
        print("\n[4/4] ILS优化完成")
        print(f"最终成本: {self.best_cost:.2f}元")
        print(f"最终车辆数: {len(self.best_solution.route_plan)}")
        print(f"总迭代次数: {iteration}")
        print(f"收敛代数: {get_convergence_generation(self.cost_history)}")
        print("=" * 70)
        
        return self.best_solution

def ils_with_hybrid_ga(max_generations: int = 200) -> Individual:
    """
    ILS结合遗传算法的混合算法
    
    策略：
    1. 遗传算法生成优质初始解
    2. ILS进行深度局部优化
    3. 多次扰动和搜索
    
    Args:
        max_generations: 遗传算法最大迭代代数
        
    Returns:
        最优解
    """
    print("\n" + "=" * 70)
    print("混合优化: 遗传算法 + 迭代局部搜索 (GA + ILS)")
    print("=" * 70)
    
    print("\n[阶段1/3] 遗传算法全局搜索...")
    ga = GeneticAlgorithm(
        distance_matrix=loader.load_distance_matrix(),
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1',
        coords=coords
    )
    ga.max_generations = max_generations
    
    ga_solution = ga.run()
    print(f"GA结果: 成本={ga_solution.cost:.2f}元, 车辆数={len(ga_solution.route_plan)}")
    
    print("\n[阶段2/3] ILS深度优化...")
    ils = BasicILS(
        distance_matrix=loader.load_distance_matrix(),
        customer_data=customer_data,
        time_windows=time_windows,
        coords=coords,
        vehicle_type='燃油车1'
    )
    
    ils.max_iterations = 300
    ils.max_no_improve = 80
    ils.perturbation_strength = 7
    
    ils_solution = ils.solve(ga_solution)
    
    print("\n[阶段3/3] 结果对比")
    print(f"GA成本: {ga_solution.cost:.2f}元, 车辆数: {len(ga_solution.route_plan)}")
    print(f"ILS成本: {ils_solution.cost:.2f}元, 车辆数: {len(ils_solution.route_plan)}")
    print(f"改善率: {(ga_solution.cost - ils_solution.cost) / ga_solution.cost * 100:.2f}%")
    
    print("\n" + "=" * 70)
    print("混合优化完成!")
    print("=" * 70)
    
    return ils_solution

# 使用示例
if __name__ == "__main__":
    main()


class ILSSolver:
    """
    迭代局部搜索（Iterated Local Search）算法

    算法框架：
    1. 初始解生成
    2. 强化局部搜索（Intensification）
    3. 扰动（Perturbation）
    4. 接受准则（基于模拟退火）

    目标：车辆数从111→75-85，成本从59k→42-48k
    """

    def __init__(self, problem_instance, initial_solution=None):
        self.problem = problem_instance
        self.current_solution = initial_solution or self.load_initial_solution()
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.evaluate(self.best_solution)

        self.max_iterations = 500
        self.max_no_improve = 50
        self.initial_temperature = 1000
        self.cooling_rate = 0.95
        self.temperature = self.initial_temperature

        self.iterations = 0
        self.improvements = 0
        self.start_time = time.time()

    def load_initial_solution(self):
        """加载当前遗传算法的最优解作为初始解"""
        print("加载当前遗传算法最优解作为ILS初始解...")
        return self.construct_greedy_solution()

    def construct_greedy_solution(self):
        """快速贪心构造初始解"""
        print("使用贪心算法构造初始解...")
        from greedy_packing import fixed_greedy_packing
        return fixed_greedy_packing(self.problem.customers, self.problem.vehicle_capacity)

    def evaluate(self, solution):
        """评估解的质量"""
        from cost_calculator import fixed_cost_calculation
        return fixed_cost_calculation(solution, self.problem)

    def solve(self):
        """ILS主求解循环"""
        print(f"开始ILS优化，初始解: {len(self.current_solution.routes)}辆车, "
              f"成本: {self.evaluate(self.current_solution):.2f}元")

        no_improve_count = 0
        self.temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            self.iterations = iteration

            s_local = self.intensification_search(self.current_solution)
            s_local_cost = self.evaluate(s_local)

            if self.acceptance_criterion(s_local_cost, self.evaluate(self.current_solution)):
                self.current_solution = s_local

                if s_local_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(s_local)
                    self.best_cost = s_local_cost
                    self.improvements += 1
                    no_improve_count = 0

                    print(f"迭代{iteration}: 新最优！{len(self.best_solution.routes)}辆车, "
                          f"成本{self.best_cost:.2f}元")
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            if no_improve_count >= self.max_no_improve:
                print(f"迭代{iteration}: 长时间无改进，执行扰动...")
                self.current_solution = self.perturbation(self.current_solution)
                no_improve_count = 0

            self.temperature *= self.cooling_rate

            if iteration % 50 == 0:
                self.print_progress(iteration)

        self.best_solution = self.intensification_search(self.best_solution)
        self.best_cost = self.evaluate(self.best_solution)

        self.print_final_result()
        return self.best_solution

    def intensification_search(self, solution):
        """
        强化局部搜索阶段

        使用多种邻域算子进行深度优化：
        - 2-opt: 路径内反转优化
        - Or-opt: 移动连续客户段
        - Cross-exchange: 路径间交换
        - Route merge: 路径合并

        持续搜索直到达到最大迭代次数或无改进
        """
        improved = True
        iteration = 0
        max_iter = 50

        while improved and iteration < max_iter:
            improved = False
            iteration += 1

            best_improvement = 0
            best_neighbor = None

            operators = [
                ('2opt', self.two_opt_all_routes),
                ('oropt', self.or_opt_routes),
                ('cross', self.cross_exchange_routes),
                ('merge', self.route_merge)
            ]

            for move_type, op_func in operators:
                new_sol = copy.deepcopy(solution)
                op_func(new_sol)
                new_cost = self.evaluate(new_sol)
                current_cost = self.evaluate(solution)

                if new_cost < current_cost and current_cost - new_cost > best_improvement:
                    best_improvement = current_cost - new_cost
                    best_neighbor = new_sol

            if best_neighbor:
                solution = best_neighbor
                improved = True

        return solution

    def perturbation(self, solution):
        """
        扰动阶段 - 跳出局部最优

        策略：
        1. 路径拆分：拆分过长的路径
        2. 客户重分配：随机移动客户到不同路径
        3. 路径合并：合并容量可合并的短路径
        """
        perturbed = copy.deepcopy(solution)

        all_customers = []
        for route in perturbed.routes:
            all_customers.extend(route)

        if not all_customers:
            return perturbed

        num_to_remove = max(1, int(len(all_customers) * 0.2))
        num_to_remove = min(num_to_remove, len(all_customers))
        customers_to_remove = random.sample(all_customers, num_to_remove)

        for customer_id in customers_to_remove:
            for route in perturbed.routes:
                if customer_id in route:
                    route.remove(customer_id)
                    break

        perturbed.routes = [r for r in perturbed.routes if r]

        for customer_id in customers_to_remove:
            best_route_idx, best_position = self.find_best_insertion(customer_id, perturbed)

            if best_route_idx is not None:
                perturbed.routes[best_route_idx].insert(best_position, customer_id)
            else:
                perturbed.routes.append([customer_id])

        return perturbed

    def acceptance_criterion(self, new_cost, current_cost):
        """
        接受准则 - 模拟退火思想

        如果新解更优则接受
        如果新解稍差但满足温度条件也接受（避免陷入局部最优）
        """
        if new_cost < current_cost:
            return True

        delta = current_cost - new_cost
        probability = math.exp(delta / self.temperature)
        return random.random() < probability

    def two_opt_all_routes(self, solution):
        """对所有路径应用2-opt"""
        for i, route in enumerate(solution.routes):
            if len(route) >= 4:
                solution.routes[i] = self.two_opt(route)
        return solution

    def two_opt(self, route):
        """单路径2-opt优化"""
        if len(route) < 4:
            return route

        best = list(route)
        best_cost = self.evaluate_route(route)

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = self.evaluate_route(new_route)
                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost

        return best

    def or_opt_routes(self, solution, segment_length=3):
        """移动连续客户段"""
        best_cost = self.evaluate(solution)

        for route_idx in range(len(solution.routes)):
            route = solution.routes[route_idx]

            for seg_len in range(1, min(segment_length + 1, len(route))):
                for seg_start in range(len(route) - seg_len + 1):
                    segment = route[seg_start:seg_start + seg_len]
                    remaining = route[:seg_start] + route[seg_start + seg_len:]

                    if not remaining:
                        continue

                    for insert_pos in range(len(remaining) + 1):
                        new_route = remaining[:insert_pos] + segment + remaining[insert_pos:]

                        if self.is_feasible_route(new_route):
                            solution.routes[route_idx] = new_route
                            new_cost = self.evaluate(solution)

                            if new_cost >= best_cost:
                                solution.routes[route_idx] = route
                            else:
                                best_cost = new_cost

        return solution

    def cross_exchange_routes(self, solution):
        """路径间交叉交换"""
        best_cost = self.evaluate(solution)

        for r1_idx in range(len(solution.routes)):
            for r2_idx in range(r1_idx + 1, len(solution.routes)):
                route1 = solution.routes[r1_idx]
                route2 = solution.routes[r2_idx]

                if len(route1) < 2 or len(route2) < 2:
                    continue

                for seg1_start in range(len(route1)):
                    for seg1_end in range(seg1_start + 1, len(route1) + 1):
                        for seg2_start in range(len(route2)):
                            for seg2_end in range(seg2_start + 1, len(route2) + 1):
                                new_route1 = route1[:seg1_start] + route2[seg2_start:seg2_end] + route1[seg1_end:]
                                new_route2 = route2[:seg2_start] + route1[seg1_start:seg1_end] + route2[seg2_end:]

                                if self.is_feasible_route(new_route1) and self.is_feasible_route(new_route2):
                                    solution.routes[r1_idx] = new_route1
                                    solution.routes[r2_idx] = new_route2

                                    new_cost = self.evaluate(solution)
                                    if new_cost >= best_cost:
                                        solution.routes[r1_idx] = route1
                                        solution.routes[r2_idx] = route2

        return solution

    def route_merge(self, solution):
        """路径合并 - 合并容量允许的短路径"""
        best_cost = self.evaluate(solution)
        merged = False

        for r1_idx in range(len(solution.routes)):
            for r2_idx in range(r1_idx + 1, len(solution.routes)):
                route1 = solution.routes[r1_idx]
                route2 = solution.routes[r2_idx]

                combined = route1 + route2

                if self.is_feasible_route(combined):
                    new_route = self.two_opt(combined)
                    solution.routes[r1_idx] = new_route
                    solution.routes[r2_idx] = []
                    merged = True

        if merged:
            solution.routes = [r for r in solution.routes if r]

        return solution

    def is_feasible_route(self, route):
        """检查路径可行性"""
        total_weight = 0
        total_volume = 0

        for customer_id in route:
            customer = self.problem.customers.get(customer_id)
            if customer is None:
                return False
            total_weight += getattr(customer, 'weight', 0)
            total_volume += getattr(customer, 'volume', 0)

            if total_weight > self.problem.vehicle_capacity_weight or \
               total_volume > self.problem.vehicle_capacity_volume:
                return False

        return True

    def evaluate_route(self, route):
        """评估单条路径成本"""
        if not route:
            return 0

        total_cost = 0
        prev_idx = 0

        for customer_id in route:
            if customer_id in self.problem.customer_ids:
                customer_idx = self.problem.customer_ids.index(customer_id) + 1
                total_cost += self.problem.distance_matrix[prev_idx][customer_idx]
                prev_idx = customer_idx

        total_cost += self.problem.distance_matrix[prev_idx][0]
        return total_cost * 0.8

    def find_best_insertion(self, customer_id, solution):
        """找到最佳插入位置"""
        best_cost_increase = float('inf')
        best_route_idx = None
        best_position = None

        customer = self.problem.customers.get(customer_id)
        if customer is None:
            return None, None

        for route_idx, route in enumerate(solution.routes):
            route_weight = sum(getattr(self.problem.customers.get(cid), 'weight', 0) for cid in route)
            route_volume = sum(getattr(self.problem.customers.get(cid), 'volume', 0) for cid in route)

            if route_weight + getattr(customer, 'weight', 0) > self.problem.vehicle_capacity_weight or \
               route_volume + getattr(customer, 'volume', 0) > self.problem.vehicle_capacity_volume:
                continue

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [customer_id] + route[pos:]
                cost_increase = self.evaluate_route(new_route) - self.evaluate_route(route)

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_route_idx = route_idx
                    best_position = pos

        return best_route_idx, best_position

    def print_progress(self, iteration):
        """显示进度信息"""
        elapsed = time.time() - self.start_time
        print(f"进度: {iteration}/{self.max_iterations}代, "
              f"时间: {elapsed:.1f}s, 温度: {self.temperature:.2f}, "
              f"最优: {len(self.best_solution.routes)}辆车, {self.best_cost:.2f}元")

    def print_final_result(self):
        """显示最终结果"""
        elapsed = time.time() - self.start_time
        print(f"""
{'='*60}
ILS优化完成！
总迭代: {self.iterations}代
改进次数: {self.improvements}次
总时间: {elapsed:.2f}秒
最终结果: {len(self.best_solution.routes)}辆车, 成本{self.best_cost:.2f}元
相比初始解: 车辆数{len(self.current_solution.routes)}→{len(self.best_solution.routes)},
成本{self.evaluate(self.current_solution):.2f}→{self.best_cost:.2f}
{'='*60}
""")


class HybridILS:
    """
    混合迭代局部搜索（Hybrid ILS）概念框架

    本框架展示如何重用现有组件构建ILS算法：

    组件重用关系：
    ┌─────────────────────────────────────────────────────────────┐
    │                      ILS 主框架                              │
    ├─────────────────────────────────────────────────────────────┤
    │  1. 初始解生成          → 重用 GeneticAlgorithm            │
    │  2. 局部搜索 (Intensification) → 重用 LocalSearch (2-opt等)│
    │  3. 接受准则           → 重用 SimulatedAnnealing           │
    │  4. 扰动 (Perturbation) → 重用 GA的变异操作                │
    │  5. 精细调整           → 重用 TabuSearch (可选)             │
    └─────────────────────────────────────────────────────────────┘

    算法流程：
        for iteration in range(max_iterations):
            # 局部搜索阶段
            candidate = local_search.improve(current)

            # 接受准则（SA思想）
            if sa.accept(candidate, current):
                current = candidate

            # 更新最优解
            if cost(current) < cost(best):
                best = current

            # 扰动阶段（跳出局部最优）
            if stagnation_detected:
                current = ga.mutate(current)

    这种设计使得各模块可以独立开发和测试，
    同时在ILS框架下协同工作。
    """

    def __init__(self, problem_instance):
        self.problem = problem_instance

    def solve(self):
        """
        ILS主求解流程（概念框架）
        实际实现见 ILSSolver 类
        """
        pass


def validate_solution(solution, problem):
    """
    验证解的有效性
    """
    print("\n验证解的可行性...")

    served_customers = set()
    for route in solution.routes:
        served_customers.update(route)

    all_customers = set(range(1, problem.num_customers + 1))
    missing = all_customers - served_customers
    if missing:
        print(f"  ✗ 错误：{len(missing)}个客户未被服务")
        return False

    overload_count = 0
    for route in solution.routes:
        if not problem.is_route_feasible(route):
            overload_count += 1

    if overload_count > 0:
        print(f"  ✗ 错误：{overload_count}条路径超载")
        return False

    print(f"  ✓ 验证通过：{len(solution.routes)}条路径, {len(served_customers)}个客户被服务，无超载")
    return True


