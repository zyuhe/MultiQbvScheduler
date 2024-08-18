from src.mqbv.parser import *
from src.mqbv.solver import *
from src.mqbv.conf_generator import *
from src.mqbv.plot import *

def mcompute(topology: Topology, mstreams: list[MStream]):
    mapping_file_path1 = "src/config/simu_topo2trdp.yaml"
    mapping_file_path2 = "src/config/simu_multicastId2ip.yaml"
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    topology_path = "src/config/topology_config.yaml"
    streams_path = "src/config/stream_config.yaml"
    mapping_file_path = "src/config/simu_topo2trdp.yaml"

    topology = topology_parser(topology_path)
    mstreams = mstream_parser(streams_path)

    # compute(topo, streams)
    mcompute(topology, mstreams)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
