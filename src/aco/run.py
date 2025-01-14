'''
-*- coding: utf-8 -*-
@Time    :   2024/12/23 11:03
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   run.py
'''
import numpy as np

from common.parser import *
from src.aco.StreamGraph import StreamGraph
from src.aco.Aco import Aco

def plot_latency_over_iterations(best_latency_history):
    plt.plot(best_latency_history, color='green', linewidth=2)
    plt.title("Trip Latency Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Latency")
    plt.show()

def aco_solve(topology, mstreams):
    distances = np.ones((len(mstreams), len(mstreams)))
    np.fill_diagonal(distances, 0)
    streamGraph = StreamGraph(mstreams, distances)
    aco = Aco(streamGraph, topology, num_ants=10,num_iterations=5)
    best_path, best_bandwidth = aco.run()
    print("best latency history：", aco.best_latency_history)
    print(best_path)
    # plot_latency_over_iterations(aco.best_latency_history)

if __name__ == '__main__':
    print(__name__)
    topology_path = "config/topology_config.yaml"
    streams_path = "config/stream_config.yaml"
    mapping_file_path = "config/simu_topo2trdp.yaml"

    topology = topology_parser(topology_path)
    mstreams = mstream_parser(streams_path)

    # compute(topo, streams)
    aco_solve(topology, mstreams)
