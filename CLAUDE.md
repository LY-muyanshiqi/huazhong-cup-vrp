# CLAUDE.md — huazhong-cup-vrp：华中杯 VRP 车辆调度

## 项目简介
华中杯数学建模 A 题：城市绿色物流配送车辆路径问题 (VRP)。Hybrid-ILS 迭代局部搜索算法求解三问题递进模型。

## 技术栈
- Python 3.8+ · numpy · pandas · matplotlib · openpyxl · scipy
- 算法：遗传算法 (GA) + 迭代局部搜索 (ILS) + 2-opt 优化
- CI/CD：GitHub Actions (flake8 + compileall)

## 项目结构
```
问题一_优化版.py           # GA 基础框架 (Config, DataLoader, Individual, etc.)
问题二.py                  # 环保政策约束 + 新能源车调度
问题三.py                  # 动态事件实时调度
配置板块.py                # 集中配置 (车辆/成本/速度参数)
综合可视化生成器.py         # 16 张综合图表生成
产出/code/code/            # ILS 算法代码 (ils_solver, local_search, perturbation)
数据/                      # Excel 数据文件
文献/                      # 参考文献 PDF
```

## 关键术语
- **VRP (Vehicle Routing Problem)**: 车辆路径问题
- **绿色配送区**: 半径 10km 的城市中心区域，燃油车限行 8:00-16:00
- **新能源车 / 燃油车**: 两种车型，不同载重和碳排放系数
- **时间窗 (Time Window)**: 客户可接受配送的时间范围
- **ILS (Iterated Local Search)**: 迭代局部搜索元启发式算法

## 开发命令
```bash
pip install -r requirements.txt
python 问题一_优化版.py     # 问题1：基础路径优化
python 问题二.py            # 问题2：政策约束
python 问题三.py            # 问题3：动态调度
python 综合可视化生成器.py   # 生成 16 张图表
```

## 注意事项
- 问题一_优化版.py 是其他模块的基础，需先确保其可运行
- 数据文件路径假定在 `数据/` 目录下
- 产出/code/code/ 为竞赛提交版代码，与根目录文件互补不重复
