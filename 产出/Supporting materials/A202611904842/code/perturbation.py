import copy
import random

class Perturbation:
    """
    扰动算子：跳出局部最优
    """

    def __init__(self, problem_instance, strength=0.2):
        self.problem = problem_instance
        self.strength = strength

    def basic_perturb(self, solution):
        """
        基础扰动：随机移除部分客户，然后重插入
        """
        perturbed = copy.deepcopy(solution)

        all_customers = []
        for route in perturbed.routes:
            all_customers.extend(route)

        if not all_customers:
            return perturbed

        num_to_remove = max(1, int(len(all_customers) * self.strength))
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

    def find_best_insertion(self, customer_id, solution):
        """
        找到最佳插入位置（最小成本增加）
        """
        best_cost_increase = float('inf')
        best_route_idx = None
        best_position = None

        customer = self.problem.customers.get(customer_id)
        if customer is None:
            return None, None

        for route_idx, route in enumerate(solution.routes):
            route_weight = sum(
                getattr(self.problem.customers.get(cid), 'weight', 0)
                for cid in route
            )
            route_volume = sum(
                getattr(self.problem.customers.get(cid), 'volume', 0)
                for cid in route
            )

            if route_weight + getattr(customer, 'weight', 0) > self.problem.vehicle_capacity_weight or \
               route_volume + getattr(customer, 'volume', 0) > self.problem.vehicle_capacity_volume:
                continue

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [customer_id] + route[pos:]
                cost_increase = self.calculate_insertion_cost(route, new_route, customer_id)

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_route_idx = route_idx
                    best_position = pos

        return best_route_idx, best_position

    def calculate_insertion_cost(self, old_route, new_route, inserted_customer):
        """
        计算插入客户带来的成本增加
        """
        old_cost = self.evaluate_route(old_route) if old_route else 0
        new_cost = self.evaluate_route(new_route) if new_route else 0
        return new_cost - old_cost

    def evaluate_route(self, route):
        """
        评估单条路径的成本
        """
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