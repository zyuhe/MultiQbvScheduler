'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 13:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   main.py
'''
import numpy as np

from src.smt.solver import *
from common.conf_generator import *
from common.plot import *

def mcompute(topology: TopologyBase, mstreams: List[MStream]):
    mapping_file_path1 = "config/simu_topo2trdp.yaml"
    mapping_file_path2 = "config/simu_multicastId2ip.yaml"
    output_xml_path_pre = "output/traffic_config_tmp_"
    qbv_solver = Solver()
    streams, res = seg_mstreams(topology, mstreams)
    if res is False:
        return
    compute_stream_omega(qbv_solver, topology, streams)
    constraints_constructor(qbv_solver, topology, streams)
    solution = constrains_solver(qbv_solver)
    if solution is not None:
        port_timelines = parse_solution_topo(solution, topology)
        stream_timelines = parse_solution_stream(solution, streams)
        hyper_period = compute_hyper_period(mstreams)
        visualize_timeline(port_timelines, hyper_period)
        visualize_timeline(stream_timelines, hyper_period)
        sanone_sw_converse_instruction(port_timelines, hyper_period)
        turn_stream_info_to_trdp_config_xml(streams, topology, mapping_file_path1, mapping_file_path2, output_xml_path_pre)

def plot_latency_over_iterations(best_latency_history):
    plt.plot(best_latency_history, color='green', linewidth=2)
    plt.title("Trip Latency Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Latency")
    plt.show()

def aco_solve(topology, mstreams):
    from src.aco.Aco import Aco
    from src.aco.StreamGraph import StreamGraph

    distances = np.ones((len(mstreams), len(mstreams)))
    for dis in distances:
        for d in range(len(dis)):
            dis[d] = 100000 / mstreams[d].size
    # print(distances)
    np.fill_diagonal(distances, 0)
    streamGraph = StreamGraph(mstreams, distances)
    aco = Aco(streamGraph, topology, num_ants=10,num_iterations=70)
    best_path, best_bandwidth = aco.run()
    print(aco.streamGraph.pheromones)
    print("best latency history：", aco.best_latency_history)
    # print("best path history：", aco.best_path_latency_history)
    print(best_path)
    plot_latency_over_iterations(aco.best_latency_history)

def ga_solve(topology, mstreams):
    from src.ga.ga import GA

    ga = GA(topology, mstreams)
    ga.run()
    print(ga.best_latency_history)
    print("best latency", ga.best_latency_history[len(ga.best_latency_history)-1])
    print("best path", ga.best_path)
    plot_latency_over_iterations(ga.best_latency_history)

def sa_solve(topology, mstreams):
    from src.sa.sa import SA

    sa = SA(topology, mstreams)
    sa.run()
    # print(sa.best_latency_history)
    print("best latency", sa.best_latency)
    print("best path", sa.best_path)
    plot_latency_over_iterations(sa.best_latency_history)

def qlearning_solve(topology, mstreams):
    from src.qlearning.qlearning import QLearning
    # 4305600
    ql = QLearning(topology, mstreams, alpha=0.01, gamma=0.8, epsilon=0.5, final_epsilon=0.05)
    ql.Train_Qtable(iter_num=2000)
    # 保存Q表
    ql.Write_Qtable()

def dqn_solve(topology, mstreams):
    from src.dqn.agent import DQN
    # 4305600
    dqn = DQN(topology, mstreams)
    dqn.Train_Qtable(iter_num=2000)
    plot_latency_over_iterations(dqn.best_latency_history)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    topology_path = "config/topology_config.yaml"
    streams_path = "config/stream_config2.yaml"
    mapping_file_path = "config/simu_topo2trdp.yaml"

    topology = topology_parser(topology_path)
    mstreams = mstream_parser(streams_path)

    # compute(topo, streams) smt
    # mcompute(topology, mstreams)

    # aco 蚁群算法 No.3
    # aco_solve(topology, mstreams)

    # ga 遗传算法 No.2
    # ga_solve(topology, mstreams)

    # sa 模拟退火 No.1 [3, 9, 4, 10, 7, 11, 8, 5, 12, 13, 14, 1, 0, 15, 6, 2]
    # sa_solve(topology, mstreams)

    # q-learning
    # qlearning_solve(topology, mstreams)

    # dqn slove
    dqn_solve(topology, mstreams)

    # calc_total_latency(topology, mstreams, [1, 3, 10, 4, 9, 11, 7, 8, 5, 14, 13, 12, 2, 6, 15, 0])
