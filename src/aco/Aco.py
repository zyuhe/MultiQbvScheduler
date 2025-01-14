'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 14:01
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   Aco.py
'''
import numpy as np

from src.aco.Ant import Ant

class Aco(object):
    def __init__(self, streamGraph, topology, num_ants, num_iterations, decay=0.5, alpha=1):
        self.streamGraph = streamGraph
        self.topology = topology
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.best_latency_history = []
        self.best_path_latency_history = []

    def update_pheromones(self, ants):
        self.streamGraph.pheromones *= self.decay
        for ant in ants:
            for i in range(len(ant.path) - 2):
                pre_stream = ant.path[i]
                next_stream = ant.path[i + 1]
                # todo: delete
                if (ant.total_latency == 0):
                    print("error get ant's total latency")
                self.streamGraph.pheromones[pre_stream][next_stream] += (
                        self.alpha / ((ant.total_latency - 4000000) / 1000))
        # print(self.streamGraph.pheromones)

    def update_stream_and_topology_winInfo(self):
        self.streamGraph.clear_mstreams_winInfo()
        self.topology.clear_all_nodes_winInfo()

    def run(self):
        best_path = None
        best_latency = np.inf
        print("num iterations: ", self.num_iterations, "num ants: ", self.num_ants)
        for i in range(self.num_iterations):
            # best_latency = np.inf
            print("iteration {} ".format(i))
            ants = [Ant(self.streamGraph, self.topology) for _ in range(self.num_ants)]
            for ant in ants:
                if ant.complete_solution() and ant.total_latency < best_latency:
                    best_path = ant.path
                    best_latency = ant.total_latency
                self.update_stream_and_topology_winInfo()
            self.update_pheromones(ants)
            self.best_latency_history.append(best_latency)
            self.best_path_latency_history.append(best_path)
        return best_path, best_latency
