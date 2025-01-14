'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 12:07
@Author  :   zyh
@Email   :
@Project :   MultiQbvScheduler
@File    :   acoExample.py
'''

'''
-*- coding: utf-8 -*-
@Time    :   2024/12/20 01:56
@Author  :   zyh
@Email   :
@Project :   qbvscheduler-web
@File    :   acoExam.py
'''

import numpy as np
import matplotlib.pyplot as plt


# Graph类代表蚂蚁将旅行的环境
class Graph:
    def __init__(self, distances):
        # 使用距离矩阵（节点之间的距离）初始化图
        self.distances = distances
        self.num_nodes = len(distances)  # 节点数（城市）
        # 初始化每条路径之间的信息素（与距离大小相同）
        self.pheromones = np.ones_like(distances, dtype=float)  # 以相等的信息素开始


# Ant类代表在图上旅行的单个蚂蚁
class Ant:
    def __init__(self, graph):
        self.graph = graph
        # 为蚂蚁选择一个随机起始节点
        self.current_node = np.random.randint(graph.num_nodes)
        self.path = [self.current_node]  # 用初始节点开始路径
        self.total_distance = 0  # 从零开始旅行距离
        # 未访问的节点是除了起始节点之外的所有节点
        self.unvisited_nodes = set(range(graph.num_nodes)) - {self.current_node}

    # 根据信息素和距离为蚂蚁选择下一个节点
    def select_next_node(self):
        # 初始化一个数组来存储每个节点的概率
        probabilities = np.zeros(self.graph.num_nodes)
        # 对于每个未访问的节点，根据信息素和距离计算概率
        for node in self.unvisited_nodes:
            if self.graph.distances[self.current_node][node] > 0:  # 只考虑可达的节点
                # 信息素越多，距离越短，节点被选择的可能性就越大
                probabilities[node] = (self.graph.pheromones[self.current_node][node] ** 2 /
                                       self.graph.distances[self.current_node][node])
                # probabilities[node] = (self.graph.pheromones[self.current_node][node] ** 2 /
                #                        50)
        probabilities /= probabilities.sum()  # 归一化概率使其总和为1
        # 根据计算出的概率选择下一个节点
        next_node = np.random.choice(range(self.graph.num_nodes), p=probabilities)
        return next_node

    # 移动到下一个节点并更新蚂蚁的路径
    def move(self):
        next_node = self.select_next_node()  # 选择下一个节点
        self.path.append(next_node)  # 添加到路径
        # 将当前节点和下一个节点之间的距离加到总距离上
        self.total_distance += self.graph.distances[self.current_node][next_node]
        self.current_node = next_node  # 更新当前节点为下一个节点
        self.unvisited_nodes.remove(next_node)  # 将下一个节点标记为已访问

    # 通过访问所有节点并返回起始节点来完成路径
    def complete_path(self):
        while self.unvisited_nodes:  # 当仍有未访问的节点时
            self.move()  # 继续移动到下一个节点
        # 在访问了所有节点后，返回起始节点以完成循环
        self.total_distance += self.graph.distances[self.current_node][self.path[0]]
        self.path.append(self.path[0])  # 在路径的末尾添加起始节点


# ACO（蚁群优化）类运行算法以找到最佳路径
class ACO:
    def __init__(self, graph, num_ants, num_iterations, decay=0.5, alpha=1.0):
        self.graph = graph
        self.num_ants = num_ants  # 每次迭代中的蚂蚁数量
        self.num_iterations = num_iterations  # 迭代次数
        self.decay = decay  # 信息素蒸发的速率
        self.alpha = alpha  # 信息素更新的强度
        self.best_distance_history = []  # 存储每次迭代中找到的最佳距离

    # 运行ACO算法的主要函数
    def run(self):
        best_path = None
        # best_distance = np.inf  # 以一个非常大的数字开始比较
        # 运行指定次数的迭代算法
        for _ in range(self.num_iterations):
            best_distance = np.inf
            ants = [Ant(self.graph) for _ in range(self.num_ants)]  # 创建一组蚂蚁
            for ant in ants:
                ant.complete_path()  # 让每只蚂蚁完成其路径
                # 如果当前蚂蚁的路径比迄今为止找到的最佳路径短，则更新最佳路径
                if ant.total_distance < best_distance:
                    best_path = ant.path
                    best_distance = ant.total_distance
            self.update_pheromones(ants)  # 根据蚂蚁的路径更新信息素
            self.best_distance_history.append(best_distance)  # 保存每次迭代的最佳距离
        return best_path, best_distance

    # 在所有蚂蚁完成旅行后更新路径上的信息素
    def update_pheromones(self, ants):
        self.graph.pheromones *= self.decay  # 减少所有路径上的信息素（蒸发）
        # 对于每只蚂蚁，根据它们的路径质量增加信息素
        for ant in ants:
            for i in range(len(ant.path) - 1):
                from_node = ant.path[i]
                to_node = ant.path[i + 1]
                # 根据蚂蚁旅行的总距离成反比更新信息素
                self.graph.pheromones[from_node][to_node] += self.alpha / ant.total_distance
        print(self.graph.pheromones)


# 为20个节点的图生成随机距离
num_nodes = 16
distances = np.random.randint(1, 100, size=(num_nodes, num_nodes))  # 随机距离在1到100之间
np.fill_diagonal(distances, 0)  # 节点自身的距离为0
graph = Graph(distances)  # 使用随机距离创建图
aco = ACO(graph, num_ants=10, num_iterations=100)  # 使用10只蚂蚁和30次迭代初始化ACO
best_path, best_distance = aco.run()  # 运行ACO算法以找到最佳路径

# 打印找到的最佳路径和总距离
print(f"Best path: {best_path}")
print(f"Total distance: {best_distance}")


# 绘制最终解决方案（第一张图）- 显示蚂蚁找到的最佳路径
def plot_final_solution(distances, path):
    num_nodes = len(distances)
    # 为节点生成随机坐标，以便在2D平面上可视化它们
    coordinates = np.random.rand(num_nodes, 2) * 10
    # 将节点（城市）作为红点绘制
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red')
    # 用索引编号标记每个节点
    for i in range(num_nodes):
        plt.text(coordinates[i, 0], coordinates[i, 1], f"{i}", fontsize=10)
    # 绘制连接节点的路径（边），显示找到的最佳路径
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        plt.plot([coordinates[start, 1], coordinates[end, 0]],
                 [coordinates[start, 0], coordinates[end, 1]],
                 'blue', linewidth=1.5)
    plt.title("Final Solution: Best Path")
    plt.show()


# 绘制迭代过程中的距离（第二张图）- 显示随着时间的推移路径长度的改善
def plot_distance_over_iterations(best_distance_history):
    # 绘制每次迭代中找到的最佳距离（应该随着时间的推移而减少）
    plt.plot(best_distance_history, color='green', linewidth=2)
    plt.title("Trip Length Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.show()


# 调用绘图函数以显示结果
plot_final_solution(distances, best_path)
plot_distance_over_iterations(aco.best_distance_history)

