'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 13:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   main.py
'''
import logging
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt

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

def plot_latency_over_iterations(best_latency_history, solver, save_dir):
    x = [x for x in range(len(best_latency_history))]
    plt.plot(x, best_latency_history, 'r', label="Best Latency")
    plt.title(f"Total Latency Over Iterations {(solver)}")
    plt.xlabel("Iteration")
    plt.ylabel("Latency")
    plt.legend()
    plt.show()
    plt.savefig(f"{save_dir}/best_latency_hist_{solver}.jpg", bbox_inches='tight', dpi=300)

def aco_solve(topology, mstreams, ns_dir_path, recorder):
    from src.aco.Aco import Aco
    from src.aco.StreamGraph import StreamGraph
    recorder.info("=== solve using ACO ===")
    ts = time.time()
    distances = np.ones((len(mstreams), len(mstreams)))
    for dis in distances:
        for d in range(len(dis)):
            dis[d] = 100000 / mstreams[d].size
    np.fill_diagonal(distances, 0)
    streamGraph = StreamGraph(mstreams, distances)
    aco = Aco(streamGraph, topology, num_ants=20,num_iterations=100)
    best_path, best_latency = aco.run()
    best_path = [int(x) for x in best_path]
    te = time.time()
    recorder.info(f"best latency: {min(aco.best_latency_history)}")
    recorder.info(f"best path：{best_path}")
    recorder.info(f"run {te-ts} seconds")
    plot_latency_over_iterations(aco.best_latency_history, "aco", ns_dir_path)

def ga_solve(topology, mstreams, ns_dir_path, recorder):
    from src.ga.ga import GA
    recorder.info("=== solve using GA ===")
    ts = time.time()
    ga = GA(topology, mstreams, 150, 100)
    ga.run()
    te = time.time()
    recorder.info(f"best latency: {ga.best_latency_history[len(ga.best_latency_history)-1]}")
    recorder.info(f"best path：{ga.best_path}")
    recorder.info(f"run {te-ts} seconds")
    plot_latency_over_iterations(ga.best_latency_history, "ga", ns_dir_path)

def sa_solve(topology, mstreams, ns_dir_path, recorder):
    from src.sa.sa import SA
    recorder.info("=== solve using SA ===")
    ts = time.time()
    sa = SA(topology, mstreams, 100, 10)
    sa.run()
    te = time.time()
    recorder.info(f"best latency: {sa.best_latency}")
    recorder.info(f"best path：{sa.best_path}")
    recorder.info(f"run {te-ts} seconds")
    plot_latency_over_iterations(sa.best_latency_history, "sa", ns_dir_path)

def qlearning_solve(topology, mstreams, ns_dir_path, recorder):
    from src.qlearning.qlearning import QLearning
    recorder.info("=== solve using Q-Learning ===")
    ts = time.time()
    ql = QLearning(topology, mstreams, recorder, alpha=0.01, gamma=0.8, epsilon=0.5, final_epsilon=0.05)
    ql.Train_Qtable(iter_num=2000)
    # 保存Q表
    ql.Write_Qtable()
    te = time.time()
    recorder.info(f"run {te - ts} seconds")
    plot_latency_over_iterations(ql.best_latency_history, "q-learning", ns_dir_path)

def dqn_solve(topology, mstreams, ns_dir_path, recorder):
    from src.dqn.agent import DQN
    recorder.info("=== solve using DQN ===")
    ts = time.time()
    dqn = DQN(topology, mstreams, recorder)
    dqn.Train_Qtable(iter_num=2000)
    te = time.time()
    recorder.info(f"run {te - ts} seconds")
    plot_latency_over_iterations(dqn.best_latency_history, "dqn", ns_dir_path)

def a3c_solve(topology, mstreams, ns_dir_path, recorder):
    from src.a3c.a3c import A3C
    recorder.info("=== solve using A3C ===")
    ts = time.time()
    a3c = A3C(topology, mstreams, recorder)
    a3c.train(num_workers=3, iter_num=1000)
    te = time.time()
    recorder.info(f"run {te - ts} seconds")
    for i in range(len(a3c.workers)):
        plot_latency_over_iterations(a3c.workers[i].best_latency_history, f"a3c_worker{i}", ns_dir_path)

def init_recorder(log_dir):
    # 创建日志记录器
    recorder = logging.getLogger()
    recorder.setLevel(logging.INFO)  # 设置日志级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台日志级别

    # 创建文件处理器
    log_file = f"{log_dir}/record.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # 文件日志级别

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    recorder.addHandler(console_handler)
    recorder.addHandler(file_handler)

    return recorder

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    topology_path = "config/topology_config3.yaml"
    streams_path = "config/stream_config2.yaml"
    mapping_file_path = "config/simu_topo2trdp.yaml"

    topology = topology_parser(topology_path)
    if "data" not in os.listdir("./"):
        os.mkdir("./data/")
    current_datetime = datetime.datetime.now()
    dir_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(f"./data/{dir_datetime}")
    for n in [50, 100]:
    # for n in [10, 20, 30, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        ns_dir_path = f"./data/{dir_datetime}/{n}"
        os.mkdir(f"{ns_dir_path}")
        recorder_dir = ns_dir_path  # 设置日志保存目录
        recorder = init_recorder(recorder_dir)
        gen_streams_path = f"{ns_dir_path}/gen_{n}_stream_config_{formatted_datetime}.yaml"
        if generate_streams(n, topology, gen_streams_path):
            streams_path = gen_streams_path
            mstreams = mstream_parser(streams_path)
            # compute(topo, streams) smt
            # mcompute(topology, mstreams)
            # aco 蚁群算法
            # TODO：turn to 1 function
            # TODO：statistics, best latency, convergence time used, success rate
            aco_solve(topology, mstreams, ns_dir_path, recorder)
            # ga 遗传算法
            ga_solve(topology, mstreams, ns_dir_path, recorder)
            # sa 模拟退火
            sa_solve(topology, mstreams, ns_dir_path, recorder)
            # q-learning
            qlearning_solve(topology, mstreams, ns_dir_path, recorder)
            # dqn slove
            dqn_solve(topology, mstreams, ns_dir_path, recorder)
            # a3c solve（a2c single thread) poor convergence
            a3c_solve(topology, mstreams, ns_dir_path, recorder)
            # ll = random.sample([i for i in list(range(len(mstreams)))], len(mstreams))
            # print(ll)
            # calc_total_latency(topology, mstreams, ll)
            '''
            for node in topology.nodes:
                for port in node.ports:
                    print(node.id, port.id)
                    print(port.windowsInfo)
            for node in topology.nodes:
                for port in node.ports:
                    name = f'node_{node.id}_port_{port.id}'
                    port_timeline = []
                    for info in port.windowsInfo:
                        port_timeline.append(
                            [info[port.TS_OPEN], round(info[port.TS_OPEN] + info[port.WIN_LEN], 1), info[port.PCP], 0])
                    if len(port_timeline) > 0:
                        draw_gantt_chart(name, port_timeline, port.hyper_period)
            '''
