| 场景 | n | 成功率 | 最终模式分布 | 位置误差 中位/最大 [mm] | 姿态误差 最大 [deg] | 腰部位移 中位/最大 [rad] | 最大力矩利用率 中位/最大 | 最小 mesh 距离 最小 [mm] | 胶囊约束被违反但 mesh 通过 | 求解时间 中位/P90/最大 [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| M1_reachable_l | 10 | 10/10 (100%) | M1:10 | 0.20/0.20 | 0.11 | 0.000/0.000 | 0.15/0.23 | 15.2 | 0/10 | 0.30/0.44/0.67 |
| M1_reachable_r | 10 | 10/10 (100%) | M1:10 | 0.20/0.20 | 0.11 | 0.000/0.000 | 0.12/0.18 | 15.4 | 0/10 | 0.37/0.59/0.94 |
| M1_hard | 20 | 2/20 (10%) | M1:2 | 0.20/0.20 | 0.11 | 0.000/0.000 | 0.12/0.16 | 19.4 | 0/2 | 0.73/1.02/1.13 |
| M2_reachable | 20 | 20/20 (100%) | M2:20 | 0.20/0.20 | 0.11 | 0.011/0.171 | 0.16/0.21 | 13.7 | 0/20 | 0.46/0.69/0.84 |
| auto_waist_needed | 20 | 20/20 (100%) | M1:2, M2:18 | 0.20/0.20 | 0.11 | 0.297/0.947 | 0.30/0.51 | 7.1 | 0/20 | 1.39/1.95/2.29 |
| specific_cross_body | 3 | 3/3 (100%) | M2:3 | 0.20/0.20 | 0.11 | 0.532/0.904 | 0.39/0.45 | 5.0 | 0/3 | 1.08/1.18/1.20 |
| torque_payload_3.0kg | 10 | 10/10 (100%) | M1:9, M2:1 | 0.20/0.20 | 0.11 | 0.000/0.021 | 0.62/0.90 | 20.2 | 0/10 | 0.28/0.76/1.28 |
| torque_payload_4.5kg | 10 | 7/10 (70%) | M1:5, M2:2 | 0.20/0.20 | 0.11 | 0.000/0.247 | 0.81/0.90 | 20.2 | 0/7 | 0.75/1.18/1.28 |
| M3_separate | 20 | 20/20 (100%) | M3:20 | 0.20/0.20 | 0.11 | 0.091/0.488 | 0.23/0.46 | 5.0 | 0/20 | 0.98/1.80/3.02 |
| box_reachable | 20 | 18/20 (90%) | M1:7, M3:11 | 0.20/0.20 | 0.11 | 0.038/0.401 | 0.56/0.90 | 6.0 | 0/18 | 2.92/4.92/6.27 |
| box_random_hard | 20 | 5/20 (25%) | M1:2, M3:3 | 0.20/0.20 | 0.11 | 0.002/0.236 | 0.62/0.63 | 5.0 | 0/5 | 3.22/4.05/4.76 |
| near_hands_side_by_side | 20 | 19/20 (95%) | M3:19 | 0.20/0.20 | 0.11 | 0.060/0.416 | 0.27/0.46 | 12.6 | 14/19 | 2.36/2.85/3.18 |
| near_crossed_hard | 5 | 0/5 (0%) | - | - | - | - | - | - | - | 2.59/3.05/3.28 |
| infeasible_inside_right_hand | 3 | 0/3 (0%) | - | - | - | - | - | - | - | 1.25/1.48/1.54 |

升级(escalation)原因统计:
- auto_waist_needed: M1 -> 下一级, 原因 'no valid solution' x18
- box_random_hard: M1 -> 下一级, 原因 'no valid solution' x18
- box_random_hard: M3 -> 下一级, 原因 'no valid solution' x15
- box_reachable: M1 -> 下一级, 原因 'no valid solution' x13
- box_reachable: M3 -> 下一级, 原因 'no valid solution' x2
- specific_cross_body: M1 -> 下一级, 原因 'no valid solution' x3
- torque_payload_3.0kg: M1 -> 下一级, 原因 'no valid solution' x1
- torque_payload_4.5kg: M1 -> 下一级, 原因 'no valid solution' x4
- torque_payload_4.5kg: M1 -> 下一级, 原因 'torque utilisation' x3
- torque_payload_4.5kg: M2 -> 下一级, 原因 'no valid solution' x3

各层耗时 (每个 stage 平均, s):
- M1_reachable_l: seed=0.001, qp=0.088, nlp=0.197, validate=0.027
- M1_reachable_r: seed=0.001, qp=0.100, nlp=0.268, validate=0.026
- M1_hard: seed=0.000, qp=0.067, nlp=0.592, validate=0.022
- M2_reachable: seed=0.001, qp=0.178, nlp=0.228, validate=0.019
- auto_waist_needed: seed=0.001, qp=0.124, nlp=0.550, validate=0.021
- specific_cross_body: seed=0.000, qp=0.126, nlp=0.343, validate=0.024
- torque_payload_3.0kg: seed=0.000, qp=0.072, nlp=0.258, validate=0.017
- torque_payload_4.5kg: seed=0.001, qp=0.123, nlp=0.284, validate=0.020
- M3_separate: seed=0.001, qp=0.290, nlp=0.797, validate=0.024
- box_reachable: seed=0.002, qp=0.397, nlp=1.221, validate=0.044
- box_random_hard: seed=0.001, qp=0.271, nlp=1.327, validate=0.039
- near_hands_side_by_side: seed=0.001, qp=0.492, nlp=1.621, validate=0.034
- near_crossed_hard: seed=0.002, qp=0.817, nlp=1.673, validate=0.050
- infeasible_inside_right_hand: seed=0.001, qp=0.126, nlp=1.014, validate=0.042

初始化 IKSolver: 5.27s；预热(构建 CasADi/IPOPT NLP，按 mode/pair 集缓存): 每个进程 11.0s，单个 NLP 构建平均 1.07s

