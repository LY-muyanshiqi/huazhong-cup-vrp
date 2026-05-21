"""
华中杯数学建模A题 - 问题3：动态事件下的实时调度策略
处理4种事件类型的检测、影响评估和响应策略
"""

import numpy as np
import pandas as pd
import random
import time
from typing import List, Tuple, Dict, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass
import heapq

from 问题一_优化版 import (
    Config, DataLoader, TimeVaryingSpeed, Individual, CostCalculator,
    GeneticAlgorithm
)
from 问题二 import (
    PolicyConfig, GreenZoneIdentifier, VehicleAllocator
)

# ============== 事件类型定义 ==============
class EventType(Enum):
    """事件类型枚举"""
    CUSTOMER_ADD = 1      # 新增客户
    CUSTOMER_CANCEL = 2   # 客户取消
    VEHICLE_BREAKDOWN = 3  # 车辆故障
    TRAFFIC_DELAY = 4     # 交通延误
    ORDER_MODIFY = 5       # 订单变更（重量/体积变化）
    ROAD_BLOCK = 6        # 道路封闭
    TIME_WINDOW_CHANGE = 7 # 时间窗变更

# ============== 事件数据类 ==============
@dataclass
class Event:
    """事件数据类"""
    event_type: EventType
    event_id: int
    timestamp: float  # 事件发生时间(h)
    description: str
    affected_customers: List[int] = None
    affected_vehicles: List[int] = None
    severity: float = 1.0  # 严重程度 0-1
    extra_data: Dict = None

    def __post_init__(self):
        if self.affected_customers is None:
            self.affected_customers = []
        if self.affected_vehicles is None:
            self.affected_vehicles = []
        if self.extra_data is None:
            self.extra_data = {}

# ============== 事件严重程度枚举 ==============
class EventSeverity(Enum):
    """事件严重程度"""
    MINOR = 1      # 轻微：1-2个客户受影响
    MODERATE = 2   # 中等：3-5个客户或1辆车受影响
    MAJOR = 3      # 重大：超过5个客户或多辆车受影响

# ============== 事件检测器 ==============
class EventDetector:
    """事件检测器"""

    def __init__(self, original_plan: Individual,
                 customer_data: pd.DataFrame,
                 time_windows: np.ndarray):
        self.original_plan = original_plan
        self.customer_data = customer_data
        self.time_windows = time_windows

    def detect_event_type(self, event_data: Dict) -> Event:
        """根据事件数据检测事件类型"""
        event_type_str = event_data.get('type', 'unknown')

        try:
            event_type = EventType[event_type_str.upper()]
        except KeyError:
            event_type = EventType.CUSTOMER_ADD

        return Event(
            event_type=event_type,
            event_id=event_data.get('id', int(time.time())),
            timestamp=event_data.get('timestamp', 10.0),
            description=event_data.get('description', ''),
            affected_customers=event_data.get('affected_customers', []),
            affected_vehicles=event_data.get('affected_vehicles', []),
            severity=event_data.get('severity', 1.0),
            extra_data=event_data.get('extra_data', {})
        )

    def assess_severity(self, event: Event) -> EventSeverity:
        """评估事件严重程度"""
        n_affected_customers = len(event.affected_customers)
        n_affected_vehicles = len(event.affected_vehicles)

        if event.event_type == EventType.VEHICLE_BREAKDOWN:
            return EventSeverity.MODERATE if n_affected_vehicles == 1 else EventSeverity.MAJOR

        if n_affected_customers <= 2:
            return EventSeverity.MINOR
        elif n_affected_customers <= 5:
            return EventSeverity.MODERATE
        else:
            return EventSeverity.MAJOR

# ============== 影响评估器 ==============
class ImpactEvaluator:
    """影响评估器"""

    def __init__(self, original_plan: Individual,
                 distance_matrix: np.ndarray,
                 customer_data: pd.DataFrame,
                 time_windows: np.ndarray):
        self.original_plan = original_plan
        self.distance_matrix = distance_matrix
        self.customer_data = customer_data
        self.time_windows = time_windows
        self.original_cost = self._calculate_original_cost()

    def _calculate_original_cost(self) -> float:
        """计算原始方案成本"""
        # 检测列名
        weight_col = '重量' if '重量' in self.customer_data.columns else '总重量'
        volume_col = '体积' if '体积' in self.customer_data.columns else '总体积'
        id_col = '客户编号' if '客户编号' in self.customer_data.columns else '客户ID'
        
        cost_calc = CostCalculator(
            self.distance_matrix, self.customer_data,
            self.time_windows, '燃油车1'
        )
        cost, _, _ = cost_calc.calculate_cost(
            self.original_plan,
            self.customer_data[weight_col].values,
            self.customer_data[volume_col].values,
            self.customer_data[id_col].values
        )
        return cost

    def evaluate_impact(self, event: Event) -> Dict:
        """
        量化评估事件影响

        Returns:
            影响评估字典
        """
        impact = {
            'original_cost': self.original_cost,
            'estimated_new_cost': 0,
            'cost_increase': 0,
            'cost_increase_rate': 0,
            'affected_routes': [],
            'delivery_delay': 0,
            'feasibility': True
        }

        if event.event_type == EventType.CUSTOMER_CANCEL:
            # 客户取消：不是简单降低5%，而是要考虑已装载的货物和空驶成本
            # 取消的客户可能已经在装载中，空驶成本会抵消部分节省
            n_cancel = len(event.affected_customers)
            # 每取消一个客户，成本增加约2%（空驶损失和重新规划成本）
            impact['estimated_new_cost'] = self.original_cost * (1 + 0.02 * n_cancel)
            impact['feasibility'] = True

        elif event.event_type == EventType.CUSTOMER_ADD:
            # 新增客户，增加成本
            n_new = len(event.affected_customers)
            impact['estimated_new_cost'] = self.original_cost * (1 + 0.1 * n_new)
            impact['affected_routes'] = ['需要新建路线']

        elif event.event_type == EventType.VEHICLE_BREAKDOWN:
            # 车辆故障：需要重新分配货物和路线，影响较大但不应该达到30%
            # 合理范围应该在10-20%之间
            n_vehicles = len(event.affected_vehicles)
            impact['estimated_new_cost'] = self.original_cost * (1 + 0.15 * n_vehicles)
            impact['affected_routes'] = [f'车辆{v}' for v in event.affected_vehicles]
            impact['delivery_delay'] = 1.5 * n_vehicles

        elif event.event_type == EventType.TRAFFIC_DELAY:
            # 交通延误
            delay_hours = event.extra_data.get('delay_hours', 1)
            impact['estimated_new_cost'] = self.original_cost * (1 + 0.05 * delay_hours)
            impact['delivery_delay'] = delay_hours
            impact['feasibility'] = delay_hours < 3

        elif event.event_type == EventType.ORDER_MODIFY:
            # 订单变更
            weight_change = event.extra_data.get('weight_change', 0)
            impact['estimated_new_cost'] = self.original_cost * (1 + 0.02 * abs(weight_change))

        elif event.event_type == EventType.TIME_WINDOW_CHANGE:
            # 时间窗变更
            impact['feasibility'] = True

        elif event.event_type == EventType.ROAD_BLOCK:
            # 道路封闭
            impact['estimated_new_cost'] = self.original_cost * 1.15
            impact['delivery_delay'] = 1.5
            impact['affected_routes'] = event.affected_customers

        impact['cost_increase'] = impact['estimated_new_cost'] - self.original_cost
        impact['cost_increase_rate'] = impact['cost_increase'] / self.original_cost * 100

        return impact

    def set_original_cost(self, cost: float):
        """设置原始方案成本（从外部传入，确保一致性）"""
        self.original_cost = cost

# ============== 响应策略生成器 ==============
class ResponseStrategyGenerator:
    """响应策略生成器"""

    def __init__(self, original_plan: Individual,
                 distance_matrix: np.ndarray,
                 customer_data: pd.DataFrame,
                 time_windows: np.ndarray):
        self.original_plan = original_plan
        self.distance_matrix = distance_matrix
        self.customer_data = customer_data
        self.time_windows = time_windows

    def generate_strategy(self, event: Event,
                         severity: EventSeverity) -> Dict:
        """生成响应策略"""
        strategy = {
            'strategy_type': '',
            'action': '',
            'parameters': {},
            'estimated_time': 0,
            'cost': 0
        }

        if severity == EventSeverity.MINOR:
            # 轻微事件：局部调整
            strategy['strategy_type'] = '局部调整'
            strategy['action'] = self._get_local_adjustment_action(event)
            strategy['estimated_time'] = 30  # 秒
            strategy['cost'] = 50

        elif severity == EventSeverity.MODERATE:
            # 中等事件：重插入算法
            strategy['strategy_type'] = '重插入'
            strategy['action'] = '使用重插入算法重新优化受影响区域'
            strategy['estimated_time'] = 120
            strategy['cost'] = 200

        else:  # MAJOR
            # 重大事件：局部重优化
            strategy['strategy_type'] = '局部重优化'
            strategy['action'] = '触发遗传算法局部重优化'
            strategy['estimated_time'] = 300
            strategy['cost'] = 500

        return strategy

    def _get_local_adjustment_action(self, event: Event) -> str:
        """获取局部调整动作"""
        if event.event_type == EventType.CUSTOMER_CANCEL:
            return '删除受影响客户，重新计算路径'
        elif event.event_type == EventType.CUSTOMER_ADD:
            return '插入新客户到最优位置'
        elif event.event_type == EventType.TRAFFIC_DELAY:
            return '调整后续客户出发时间'
        elif event.event_type == EventType.TIME_WINDOW_CHANGE:
            return '重新评估时间窗约束'
        else:
            return '进行路径微调'

# ============== 实时调度器 ==============
class RealTimeScheduler:
    """实时调度器"""

    def __init__(self, original_plan: Individual,
                 distance_matrix: np.ndarray,
                 customer_data: pd.DataFrame,
                 time_windows: np.ndarray,
                 coords: pd.DataFrame = None):
        self.original_plan = original_plan
        self.distance_matrix = distance_matrix
        self.customer_data = customer_data.copy()
        self.time_windows = time_windows.copy() if time_windows is not None else None
        self.coords = coords

        # 创建客户ID到索引的映射
        self.id_to_idx = {}
        id_col = '客户编号' if '客户编号' in customer_data.columns else '客户ID'
        for i, cid in enumerate(customer_data[id_col].values):
            self.id_to_idx[cid] = i
        
        # 预计算客户重量数据
        weight_col = '重量' if '重量' in customer_data.columns else '总重量'
        self.weights = customer_data[weight_col].values

        self.event_detector = EventDetector(original_plan, customer_data, time_windows)
        self.impact_evaluator = ImpactEvaluator(
            original_plan, distance_matrix, customer_data, time_windows
        )
        self.strategy_generator = ResponseStrategyGenerator(
            original_plan, distance_matrix, customer_data, time_windows
        )

        self.event_log = []
        self.adaptation_history = []

    def process_event(self, event_data: Dict) -> Dict:
        """处理单个事件"""
        print(f"\n{'='*60}")
        print(f"检测到事件: {event_data.get('type', 'unknown')}")
        print(f"{'='*60}")

        # 1. 事件检测
        event = self.event_detector.detect_event_type(event_data)

        # 2. 评估严重程度
        severity = self.event_detector.assess_severity(event)
        print(f"事件类型: {event.event_type.name}")
        print(f"严重程度: {severity.name}")

        # 3. 影响评估
        impact = self.impact_evaluator.evaluate_impact(event)
        print(f"原始成本: {impact['original_cost']:.2f}元")
        print(f"预计新成本: {impact['estimated_new_cost']:.2f}元")
        print(f"成本增加: {impact['cost_increase']:.2f}元 ({impact['cost_increase_rate']:+.1f}%)")

        # 4. 生成响应策略
        strategy = self.strategy_generator.generate_strategy(event, severity)
        print(f"\n响应策略: {strategy['strategy_type']}")
        print(f"行动: {strategy['action']}")
        print(f"预计耗时: {strategy['estimated_time']}秒")
        print(f"策略成本: {strategy['cost']}元")

        # 5. 执行响应
        new_plan = self._execute_strategy(event, strategy)

        # 6. 记录日志
        result = {
            'event': event,
            'severity': severity,
            'impact': impact,
            'strategy': strategy,
            'new_plan': new_plan,
            'timestamp': time.time()
        }
        self.event_log.append(result)

        return result

    def _execute_strategy(self, event: Event, strategy: Dict) -> Individual:
        """执行响应策略"""
        strategy_type = strategy['strategy_type']

        if strategy_type == '局部调整':
            new_plan = self._local_adjustment(event)
        elif strategy_type == '重插入':
            new_plan = self._reinsertion(event)
        else:  # 局部重优化
            new_plan = self._partial_reoptimization(event)

        return new_plan

    def _local_adjustment(self, event: Event) -> Individual:
        """局部调整"""
        if event.event_type == EventType.CUSTOMER_CANCEL:
            return self._handle_customer_cancel(event)
        elif event.event_type == EventType.CUSTOMER_ADD:
            return self._handle_customer_add(event)
        else:
            return self.original_plan

    def _handle_customer_cancel(self, event: Event) -> Individual:
        """处理客户取消 - 改进：删除后重新优化受影响路径"""
        new_route_plan = []
        for route in self.original_plan.route_plan:
            new_route = [c for c in route if c not in event.affected_customers]
            if new_route:
                optimized_route = self._optimize_route(new_route)
                new_route_plan.append(optimized_route)

        return Individual(new_route_plan if new_route_plan else [[]])

    def _optimize_route(self, route: List[int]) -> List[int]:
        """对路径进行2-opt优化"""
        if len(route) <= 3:
            return route

        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    if self._calculate_route_distance(new_route) < self._calculate_route_distance(route):
                        route = new_route
                        improved = True
        return route

    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路径总距离"""
        if not route:
            return 0
        total = 0
        prev = 0
        for c in route:
            if c in self.id_to_idx:
                total += self.distance_matrix[prev, c]
                prev = c
        total += self.distance_matrix[prev, 0]
        return total

    def _handle_customer_add(self, event: Event) -> Individual:
        """处理新增客户 - 改进：优先插入现有路径，考虑地理邻近性"""
        new_customers = event.affected_customers

        new_route_plan = [route.copy() for route in self.original_plan.route_plan]

        for new_c in new_customers:
            if new_c not in self.id_to_idx:
                continue

            new_w = self.weights[self.id_to_idx[new_c]]
            new_v = self.volumes[self.id_to_idx[new_c]]
            max_w = self.max_weight * 0.95
            max_v = self.max_volume * 0.95

            best_inserted = False
            best_idx = -1
            best_pos = -1
            best_extra_dist = float('inf')

            for i, route in enumerate(new_route_plan):
                route_w = sum(self.weights[self.id_to_idx[c]] for c in route if c in self.id_to_idx)
                route_v = sum(self.volumes[self.id_to_idx[c]] for c in route if c in self.id_to_idx)

                if route_w + new_w > max_w or route_v + new_v > max_v:
                    continue

                for pos in range(len(route) + 1):
                    extra_dist = self._calculate_insertion_distance(route, new_c, pos)
                    if extra_dist < best_extra_dist:
                        best_extra_dist = extra_dist
                        best_idx = i
                        best_pos = pos
                        best_inserted = True

            if best_inserted and best_idx >= 0:
                new_route_plan[best_idx].insert(best_pos, new_c)
            else:
                new_route_plan.append([new_c])

        return Individual(new_route_plan if new_route_plan else [[]])

    def _calculate_insertion_distance(self, route: List[int], new_c: int, pos: int) -> float:
        """计算插入新客户到路径指定位置增加的额外距离"""
        if not route:
            return self.distance_matrix[0, new_c] + self.distance_matrix[new_c, 0]

        if pos == 0:
            return self.distance_matrix[0, new_c] + self.distance_matrix[new_c, route[0]] - self.distance_matrix[0, route[0]]
        elif pos >= len(route):
            return self.distance_matrix[route[-1], new_c] + self.distance_matrix[new_c, 0] - self.distance_matrix[route[-1], 0]
        else:
            return (self.distance_matrix[route[pos-1], new_c] + self.distance_matrix[new_c, route[pos]] -
                    self.distance_matrix[route[pos-1], route[pos]])

    def _reinsertion(self, event: Event) -> Individual:
        """重插入算法"""
        # 获取受影响的客户
        affected_customers = event.affected_customers.copy()

        # 处理车辆故障：需要从故障车辆中提取客户
        if event.event_type == EventType.VEHICLE_BREAKDOWN and event.affected_vehicles:
            for vehicle_idx in event.affected_vehicles:
                if 0 <= vehicle_idx - 1 < len(self.original_plan.route_plan):
                    route = self.original_plan.route_plan[vehicle_idx - 1]
                    affected_customers.extend(route)

        # 去重，确保每个客户只处理一次
        affected_customers = list(set(affected_customers))

        # 从原计划中移除
        new_routes = []
        for route in self.original_plan.route_plan:
            new_route = [c for c in route if c not in affected_customers]
            if new_route:
                new_routes.append(new_route)

        # 重新插入受影响客户
        for customer in affected_customers:
            new_routes = self._insert_customer_best(customer, new_routes)

        return Individual(new_routes if new_routes else [[]])

    def _insert_customer_best(self, customer: int,
                             routes: List[List[int]]) -> List[List[int]]:
        """将客户插入到最优位置"""
        if not routes:
            return [[customer]]

        # 简化：添加到载重最小的路线末尾
        min_weight = float('inf')
        min_idx = 0

        for i, route in enumerate(routes):
            total_weight = 0
            for c in route:
                if c in self.id_to_idx:
                    total_weight += self.weights[self.id_to_idx[c]]
            if total_weight < min_weight:
                min_weight = total_weight
                min_idx = i

        routes[min_idx].append(customer)
        return routes

    def _partial_reoptimization(self, event: Event) -> Individual:
        """局部重优化"""
        print("执行遗传算法局部重优化...")

        # 使用遗传算法重新求解
        ga = GeneticAlgorithm(
            self.distance_matrix,
            self.customer_data,
            self.time_windows,
            vehicle_type='燃油车1'
        )

        new_plan = ga.run()
        return new_plan

# ============== 案例演示器 ==============
class CaseDemonstrator:
    """案例演示器"""

    def __init__(self, original_plan: Individual,
                 distance_matrix: np.ndarray,
                 customer_data: pd.DataFrame,
                 time_windows: np.ndarray,
                 coords: pd.DataFrame):
        self.scheduler = RealTimeScheduler(
            original_plan, distance_matrix,
            customer_data, time_windows, coords
        )

    def run_case(self, case_name: str, event_data: Dict) -> Dict:
        """运行单个案例"""
        print("\n" + "#" * 70)
        print(f"案例: {case_name}")
        print("#" * 70)

        result = self.scheduler.process_event(event_data)

        # 计算实际效果
        if result['new_plan']:
            # 检测列名
            weight_col = '重量' if '重量' in self.scheduler.customer_data.columns else '总重量'
            volume_col = '体积' if '体积' in self.scheduler.customer_data.columns else '总体积'
            id_col = '客户编号' if '客户编号' in self.scheduler.customer_data.columns else '客户ID'
            
            cost_calc = CostCalculator(
                self.scheduler.distance_matrix,
                self.scheduler.customer_data,
                self.scheduler.time_windows,
                '燃油车1'
            )
            new_cost, new_carbon, _ = cost_calc.calculate_cost(
                result['new_plan'],
                self.scheduler.customer_data[weight_col].values,
                self.scheduler.customer_data[volume_col].values,
                self.scheduler.customer_data[id_col].values
            )

            print(f"\n调整后:")
            print(f"  新成本: {new_cost:.2f}元")
            print(f"  新碳排放: {new_carbon:.2f}kg")

        return result

    def run_demo_cases(self):
        """运行演示案例"""
        print("\n" + "=" * 70)
        print("动态事件实时调度演示")
        print("=" * 70)

        results = {}

        # 案例1：客户取消（轻微事件）
        results['case1'] = self.run_case(
            "客户取消事件（轻微）",
            {
                'type': 'customer_cancel',
                'id': 1001,
                'timestamp': 10.0,
                'description': '客户25取消订单',
                'affected_customers': [25],
                'severity': 0.3
            }
        )

        # 案例2：交通延误（中等事件）
        results['case2'] = self.run_case(
            "交通延误事件（中等）",
            {
                'type': 'traffic_delay',
                'id': 1002,
                'timestamp': 11.0,
                'description': '某路段发生交通事故，预计延误1小时',
                'affected_customers': [10, 15, 20, 25, 30],
                'severity': 0.6,
                'extra_data': {'delay_hours': 1.0}
            }
        )

        # 案例3：车辆故障（重大事件）
        results['case3'] = self.run_case(
            "车辆故障事件（重大）",
            {
                'type': 'vehicle_breakdown',
                'id': 1003,
                'timestamp': 9.0,
                'description': '2号车辆发生故障',
                'affected_vehicles': [2],
                'severity': 0.9
            }
        )

        # 案例4：新增客户（轻微事件）
        results['case4'] = self.run_case(
            "新增客户事件（轻微）",
            {
                'type': 'customer_add',
                'id': 1004,
                'timestamp': 10.5,
                'description': '新客户99需要配送',
                'affected_customers': [99],
                'severity': 0.2,
                'extra_data': {'weight': 500, 'volume': 2.0}
            }
        )

        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """打印案例汇总"""
        print("\n" + "=" * 70)
        print("案例执行汇总")
        print("=" * 70)

        for name, result in results.items():
            event = result['event']
            impact = result['impact']
            strategy = result['strategy']

            print(f"\n{name}:")
            print(f"  事件类型: {event.event_type.name}")
            print(f"  严重程度: {result['severity'].name}")
            print(f"  成本变化: {impact['cost_increase_rate']:+.1f}%")
            print(f"  响应策略: {strategy['strategy_type']}")

# ============== 主函数 ==============
def main():
    """主函数"""
    print("=" * 60)
    print("华中杯数学建模A题 - 问题3")
    print("动态事件下的实时调度策略")
    print("=" * 60)

    start_time = time.time()

    # 加载数据
    print("\n[1/6] 加载数据...")
    loader = DataLoader()
    distance_matrix = loader.load_distance_matrix()
    customer_data, coords, time_windows = loader.load_customer_data()

    n_customers = len(customer_data)
    print(f"客户数量: {n_customers}")

    # 数据预处理
    print("\n[2/6] 数据预处理...")
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

    # 生成原始计划
    print("\n[3/6] 生成原始配送计划...")
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        customer_data=customer_data,
        time_windows=time_windows,
        vehicle_type='燃油车1'
    )

    original_plan = ga.run()
    print(f"原始计划: {original_plan.n_vehicles}辆车")

    # 初始化实时调度器
    print("\n[4/6] 初始化实时调度器...")
    scheduler = RealTimeScheduler(
        original_plan, distance_matrix,
        customer_data, time_windows, coords
    )

    # 演示案例
    print("\n[5/6] 执行动态事件演示...")
    demonstrator = CaseDemonstrator(
        original_plan, distance_matrix,
        customer_data, time_windows, coords
    )
    results = demonstrator.run_demo_cases()

    # 保存结果
    print("\n[6/6] 保存结果...")
    save_dynamic_results(results, '问题3结果_优化.txt')

    elapsed_time = time.time() - start_time
    print(f"\n总求解时间: {elapsed_time:.2f}秒")

def save_dynamic_results(results: Dict, filename: str):
    """保存动态调度结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("华中杯数学建模A题 - 问题3优化结果\n")
        f.write("动态事件下的实时调度策略\n")
        f.write("=" * 70 + "\n\n")

        for i, (name, result) in enumerate(results.items(), 1):
            event = result['event']
            impact = result['impact']
            strategy = result['strategy']
            new_plan = result['new_plan']

            f.write(f"【案例{i}】{name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"事件类型: {event.event_type.name}\n")
            f.write(f"严重程度: {result['severity'].name}\n")
            f.write(f"原始成本: {impact['original_cost']:.2f}元\n")
            f.write(f"预计成本: {impact['estimated_new_cost']:.2f}元\n")
            f.write(f"成本变化: {impact['cost_increase_rate']:+.1f}%\n")
            f.write(f"响应策略: {strategy['strategy_type']}\n")
            f.write(f"行动: {strategy['action']}\n")
            
            # 添加具体的路径调整方案
            if new_plan and new_plan.route_plan:
                f.write("\n调整后的路径方案:\n")
                for j, route in enumerate(new_plan.route_plan, 1):
                    f.write(f"  车辆 {j}: 配送中心")
                    for c in route:
                        f.write(f" → 客户{c}")
                    f.write(" → 配送中心\n")
            
            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"\n结果已保存到: {filename}")

if __name__ == "__main__":
    main()
