# 计算机前沿课程论文复现源码

张柏林 2300271029

论文：Uncertainty-aware probabilistic travel time prediction for on-demand ride-hailing at didi.

实验结果

|                  | MAPE   | MAE   | RMSE   |
| ---------------- | ------ | ----- | ------ |
| WDR              | 17.76% | 95.61 | 155.79 |
| ProbTTE_cla      | 17.66% | 94.64 | 154.92 |
| ProbTTE_soft     | 17.63% | 94.35 | 154.88 |
| ProbTTE_distance | 17.33% | 93.41 | 153.48 |

源码中直接跑对应的test文件可以复现以上结果，其中text.py对应WDR，text2.py对应ProbTTE_cla和ProbTTE_soft（使用TrajDataset3代替TrajDataset2，用于生成软标签），text3.py对应ProbTTE_distance。