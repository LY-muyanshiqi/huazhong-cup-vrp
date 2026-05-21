import time
import random
import copy
import math
from typing import List, Tuple

class ILSSolver:
    """
    迭代局部搜索主框架
    目标：车辆数从111→75-85，成本从59k→42-48k
    """

    def __init__(self, problem_instance, initial_solution=None):
        """
        初始化ILS求解器

        Args:
            problem_instance: 问题实例（包含客户、车辆、距离等信息）
            initial_solution: 初始解（可选，默认使用当前遗传算法结果）
        """
        self.problem = problem_instance
        self.current_solution = initial_solution or self.load_initial_solution()
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.evaluate(self.best_solution)

        self.max_iterations = 500
        self.max_no_improve = 50
        self.perturb_strength = 0.2
        self.local_search_depth = 10

        self.iterations = 0
        self.improvements = 0
        self.start_time = time.time()

    def load_initial_solution(self):
        """加载当前遗传算法的最优解作为初始解"""
        print("加载当前遗传算法最优解作为ILS初始解...")
        return self.construct_greedy_solution()

    def construct_greedy_solution(self):
        """快速贪心构造初始解（备用）"""
        print("使用贪心算法构造初始解...")
        from greedy_packing import fixed_greedy_packing
        return fixed_greedy_packing(self.problem.customers, self.problem.vehicle_capacity)

    def evaluate(self, solution):
        """评估解的质量（成本越低越好）"""
        from cost_calculator import fixed_cost_calculation
        return fixed_cost_calculation(solution, self.problem)

    def solve(self):
        """ILS主求解循环"""
        print(f"开始ILS优化，初始解: {len(self.current_solution.routes)}辆车, "
              f"成本: {self.evaluate(self.current_solution):.2f}元")

        no_improve_count = 0

        for iteration in range(self.max_iterations):
            self.iterations = iteration

            new_solution = self.local_search(self.current_solution)
            new_cost = self.evaluate(new_solution)
            current_cost = self.evaluate(self.current_solution)

            if self.accept_solution(new_cost, current_cost, iteration):
                self.current_solution = new_solution

                if new_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(new_solution)
                    self.best_cost = new_cost
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
                self.current_solution = self.perturb(self.current_solution)
                no_improve_count = 0

            if iteration % 50 == 0:
                self.print_progress(iteration)

        self.best_solution = self.local_search(self.best_solution)
        self.best_cost = self.evaluate(self.best_solution)

        self.print_final_result()
        return self.best_solution

    def accept_solution(self, new_cost, current_cost, iteration):
        """模拟退火接受准则"""
        if new_cost < current_cost:
            return True

        temperature = self.calculate_temperature(iteration)
        probability = math.exp((current_cost - new_cost) / temperature)
        return random.random() < probability

    def calculate_temperature(self, iteration):
        """计算当前温度（退火计划）"""
        initial_temp = 1000
        cooling_rate = 0.95
        return initial_temp * (cooling_rate ** iteration)

    def local_search(self, solution):
        """局部搜索 - 需要实现具体的邻域移动算子"""
        return solution

    def perturb(self, solution):
        """扰动操作 - 需要实现具体的扰动策略"""
        return solution

    def print_progress(self, iteration):
        """显示进度信息"""
        elapsed = time.time() - self.start_time
        print(f"进度: {iteration}/{self.max_iterations}代, "
              f"时间: {elapsed:.1f}s, "
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
相比初始解: 车辆数{len(self.current_solution.routes)}→{len(self.best_solution.routes)}, "
              f"成本{self.evaluate(self.current_solution):.2f}→{self.best_cost:.2f}
{'='*60}
""")