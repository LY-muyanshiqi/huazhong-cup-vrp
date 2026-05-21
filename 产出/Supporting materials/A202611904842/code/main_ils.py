def main():
    """
    ILS主程序
    """
    print("=" * 60)
    print("迭代局部搜索(ILS) - 城市绿色物流配送调度优化")
    print("=" * 60)

    from problem_loader import load_problem_instance
    problem = load_problem_instance("data/problem1.json")

    from solution_loader import load_current_best_solution
    initial_solution = load_current_best_solution("results/genetic_best.json")

    if initial_solution is None:
        print("警告：无法加载现有解，将使用贪心构造初始解")

    from ils_solver import ILSSolver
    solver = ILSSolver(problem, initial_solution)

    print("\n开始ILS优化过程...")
    best_solution = solver.solve()

    from result_saver import save_solution
    save_solution(best_solution, "results/ils_best.json")

    validate_solution(best_solution, problem)

    print("\nILS优化完成！结果已保存到 results/ils_best.json")


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


if __name__ == "__main__":
    main()