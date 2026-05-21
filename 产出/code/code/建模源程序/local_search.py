import copy
from typing import List

class LocalSearch:
    """
    局部搜索算子集合
    先实现基础版本，确保能运行，再逐步增强
    """

    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.max_local_iterations = 50

    def basic_local_search(self, solution):
        """
        基础局部搜索：2-opt + 客户交换
        确保每次调用都有改进可能
        """
        improved = True
        iteration = 0

        while improved and iteration < self.max_local_iterations:
            improved = False
            iteration += 1

            for i, route in enumerate(solution.routes):
                if len(route) >= 3:
                    optimized = self.two_opt(route)
                    if self.evaluate_route(optimized) < self.evaluate_route(route):
                        solution.routes[i] = optimized
                        improved = True

            for i, route in enumerate(solution.routes):
                if len(route) >= 2:
                    for j in range(len(route) - 1):
                        new_route = route.copy()
                        new_route[j], new_route[j+1] = new_route[j+1], new_route[j]

                        if self.is_feasible_route(new_route) and \
                           self.evaluate_route(new_route) < self.evaluate_route(route):
                            solution.routes[i] = new_route
                            improved = True

            if len(solution.routes) >= 2:
                merged = self.try_simple_merge(solution)
                if merged and self.evaluate(merged) < self.evaluate(solution):
                    solution = merged
                    improved = True

        return solution

    def two_opt(self, route):
        """
        2-opt优化：经典路径优化算法
        """
        if len(route) < 4:
            return route

        best = list(route)
        best_cost = self.evaluate_route(route)

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue

                new_route = route[:i] + route[i:j][::-1] + route[j:]

                if self.is_feasible_route(new_route):
                    new_cost = self.evaluate_route(new_route)
                    if new_cost < best_cost:
                        best = new_route
                        best_cost = new_cost

        return best

    def try_simple_merge(self, solution):
        """
        尝试合并两条短路径
        """
        routes_with_idx = [(i, route) for i, route in enumerate(solution.routes)]
        routes_with_idx.sort(key=lambda x: len(x[1]))

        for idx1, route1 in routes_with_idx:
            for idx2, route2 in routes_with_idx:
                if idx1 >= idx2:
                    continue

                combined = route1 + route2

                if self.is_feasible_route(combined):
                    new_solution = copy.deepcopy(solution)

                    route1_len = len(route1)
                    route2_len = len(route2)

                    if route1_len <= route2_len:
                        new_solution.routes[idx1] = combined
                        new_solution.routes[idx2] = []
                    else:
                        new_solution.routes[idx2] = combined
                        new_solution.routes[idx1] = []

                    new_solution.routes = [r for r in new_solution.routes if r]

                    optimized_route = self.two_opt(combined)
                    new_solution.routes.append(optimized_route)

                    return new_solution

        return None

    def is_feasible_route(self, route):
        """
        检查路径可行性（容量约束）
        """
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
        """
        评估单条路径的成本
        """
        if not route:
            return 0

        total_cost = 0
        prev_idx = 0

        for customer_id in route:
            customer_idx = self.problem.customer_ids.index(customer_id) + 1
            total_cost += self.problem.distance_matrix[prev_idx][customer_idx]
            prev_idx = customer_idx

        total_cost += self.problem.distance_matrix[prev_idx][0]

        return total_cost * 0.8

    def evaluate(self, solution):
        """
        评估整个解的总成本
        """
        total = 0
        for route in solution.routes:
            total += self.evaluate_route(route)
        return total

    def guaranteed_improvement_local_search(self, solution):
        """
        确保每次调用都有改进的局部搜索
        尝试多种算子，选择最好的
        """
        best_solution = copy.deepcopy(solution)
        best_cost = self.evaluate(best_solution)

        operators = [
            self.two_opt_all,
            self.cross_exchange,
            self.or_opt
        ]

        for op in operators:
            new_solution = op(copy.deepcopy(solution))
            new_cost = self.evaluate(new_solution)

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

        return best_solution

    def two_opt_all(self, solution):
        """
        对所有路径应用2-opt优化
        """
        new_solution = copy.deepcopy(solution)
        for i, route in enumerate(new_solution.routes):
            if len(route) >= 4:
                new_solution.routes[i] = self.two_opt(route)
        return new_solution

    def cross_exchange(self, solution):
        """
        路径间交叉交换：两条路径交换客户段
        """
        best_solution = copy.deepcopy(solution)
        best_cost = self.evaluate(best_solution)

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
                                    new_solution = copy.deepcopy(solution)
                                    new_solution.routes[r1_idx] = new_route1
                                    new_solution.routes[r2_idx] = new_route2

                                    new_cost = self.evaluate(new_solution)
                                    if new_cost < best_cost:
                                        best_solution = new_solution
                                        best_cost = new_cost

        return best_solution

    def or_opt(self, solution, segment_length=3):
        """
        移动连续客户段：将一段连续客户移动到其他位置
        """
        best_solution = copy.deepcopy(solution)
        best_cost = self.evaluate(best_solution)

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
                            new_solution = copy.deepcopy(solution)
                            new_solution.routes[route_idx] = new_route

                            new_cost = self.evaluate(new_solution)
                            if new_cost < best_cost:
                                best_solution = new_solution
                                best_cost = new_cost

        return best_solution