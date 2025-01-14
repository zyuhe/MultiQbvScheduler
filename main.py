'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 13:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   main.py
'''
import numpy as np

from src.aco.Aco import Aco
from src.aco.StreamGraph import StreamGraph
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    topology_path = "config/topology_config.yaml"
    streams_path = "config/stream_config2.yaml"
    mapping_file_path = "config/simu_topo2trdp.yaml"

    topology = topology_parser(topology_path)
    mstreams = mstream_parser(streams_path)

    # compute(topo, streams)
    # mcompute(topology, mstreams)

    # aco
    aco_solve(topology, mstreams)
