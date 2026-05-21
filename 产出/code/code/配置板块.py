"""
华中杯数学建模A题 - 配置模块
根据题目要求更新所有参数
"""


# ============== 基础配置 ==============
class BaseConfig:
    """基础配置"""
    # 配送中心坐标
    DEPOT_COORD = (20, 20)

    # 市中心坐标（绿色配送区圆心）
    CITY_CENTER = (0, 0)

    # 绿色配送区半径（km）
    GREEN_ZONE_RADIUS = 10

    # 限行时段
    RESTRICTED_HOURS = (8, 16)  # 8:00-16:00

    # 默认数据路径
    DATA_DIR = '数据'

    # 随机种子（保证结果可复现）
    RANDOM_SEED = 42

    # 服务时间（小时）- 20分钟
    SERVICE_TIME = 20 / 60


# ============== 车辆配置 ==============
class VehicleConfig:
    """车辆配置"""

    VEHICLE_TYPES = {
        '燃油车1': {
            'capacity_kg': 3000,
            'capacity_m3': 13.5,
            'count': 60,
            'start_cost': 400,
            'fuel_type': 'fuel',
        },
        '燃油车2': {
            'capacity_kg': 1500,
            'capacity_m3': 10.8,
            'count': 50,
            'start_cost': 400,
            'fuel_type': 'fuel',
        },
        '燃油车3': {
            'capacity_kg': 1250,
            'capacity_m3': 6.5,
            'count': 50,
            'start_cost': 400,
            'fuel_type': 'fuel',
        },
        '新能源1': {
            'capacity_kg': 3000,
            'capacity_m3': 15,
            'count': 10,
            'start_cost': 400,
            'fuel_type': 'electric',
        },
        '新能源2': {
            'capacity_kg': 1250,
            'capacity_m3': 8.5,
            'count': 15,
            'start_cost': 400,
            'fuel_type': 'electric',
        },
    }

    # 默认使用车型
    DEFAULT_VEHICLE = '燃油车1'


# ============== 成本配置（题目给定） ==============
class CostConfig:
    """成本配置"""

    # 启动成本（元/辆）
    START_COST = 400

    # 能源单价
    FUEL_PRICE = 7.61  # 燃油：7.61元/L
    ELECTRIC_PRICE = 1.64  # 电费：1.64元/kWh

    # 碳排放成本（元/kg）
    CARBON_COST = 0.65

    # 碳排放转换系数
    FUEL_CARBON_FACTOR = 2.547  # η = 2.547 kg CO₂/L
    ELECTRIC_CARBON_FACTOR = 0.501  # γ = 0.501 kg CO₂/kWh

    # 时间窗惩罚（元/小时）
    WAIT_COST = 20  # 早到等待成本
    LATE_PENALTY = 50  # 晚到惩罚

    # 满载能耗增量
    FUEL_LOAD_FACTOR = 0.40  # 燃油车满载高40%
    ELECTRIC_LOAD_FACTOR = 0.35  # 新能源车满载高35%


# ============== 速度配置（题目给定） ==============
class SpeedConfig:
    """速度时变配置"""

    # 速度时段分布 N(均值, 标准差)
    SPEED_PERIODS = {
        '顺畅': {
            'time_slots': [(9, 10), (13, 15)],  # 9:00-10:00, 13:00-15:00
            'mean': 55.3,
            'std': 0.12 ** 0.5,  # 标准差
        },
        '一般': {
            'time_slots': [(10, 11.5), (15, 17)],  # 10:00-11:30, 15:00-17:00
            'mean': 35.4,
            'std': 5.22 ** 0.5,
        },
        '拥堵': {
            'time_slots': [(8, 9), (11.5, 13)],  # 8:00-9:00, 11:30-13:00
            'mean': 9.8,
            'std': 4.72 ** 0.5,
        },
    }


# ============== 能耗计算函数 ==============
def calculate_energy_cost(distance_km, vehicle_type, load_ratio=0.5):
    """
    计算运输成本和碳排放成本

    参数：
        distance_km: 行驶距离（km）
        vehicle_type: 'fuel' 或 'electric'
        load_ratio: 载重率（当前载重/最大载重）

    返回：
        (运输成本, 碳排放成本)
    """
    import numpy as np

    # 使用平均速度计算FPK/EPK
    speed = 35.4  # 取一般时段的平均速度作为基准

    if vehicle_type == 'fuel':
        # FPK = 0.0025*ν² - 0.2554*ν + 31.75
        fpk = 0.0025 * speed ** 2 - 0.2554 * speed + 31.75
        # 满载能耗高40%
        energy_per_100km = fpk * (1 + CostConfig.FUEL_LOAD_FACTOR * load_ratio)
        # 运输成本
        energy_cost = energy_per_100km * distance_km / 100 * CostConfig.FUEL_PRICE
        # 碳排放
        carbon = fpk * distance_km / 100 * CostConfig.FUEL_CARBON_FACTOR
    else:
        # EPK = 0.0014*ν² - 0.12*ν + 36.19
        epk = 0.0014 * speed ** 2 - 0.12 * speed + 36.19
        # 满载能耗高35%
        energy_per_100km = epk * (1 + CostConfig.ELECTRIC_LOAD_FACTOR * load_ratio)
        # 运输成本
        energy_cost = energy_per_100km * distance_km / 100 * CostConfig.ELECTRIC_PRICE
        # 碳排放
        carbon = epk * distance_km / 100 * CostConfig.ELECTRIC_CARBON_FACTOR

    carbon_cost = carbon * CostConfig.CARBON_COST

    return energy_cost, carbon_cost


# ============== 速度计算函数 ==============
def get_speed_at_time(hour):
    """
    根据时间获取速度

    参数：
        hour: 时间（小时，如 8.5 表示 8:30）

    返回：
        速度（km/h）
    """
    import numpy as np

    periods = SpeedConfig.SPEED_PERIODS

    for period_name, period_info in periods.items():
        for start, end in period_info['time_slots']:
            if start <= hour < end:
                mean = period_info['mean']
                std = period_info['std']
                speed = np.random.normal(mean, std)
                return max(speed, 5)  # 保证速度至少为5

    # 默认返回一般时段
    return 35.4


def get_average_speed(hour_range=(9, 17)):
    """
    获取时间区间的平均速度
    """
    import numpy as np
    hours = np.arange(hour_range[0], hour_range[1], 0.5)
    speeds = [get_speed_at_time(h) for h in hours]
    return np.mean(speeds)
