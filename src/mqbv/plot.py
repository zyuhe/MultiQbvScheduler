'''
-*- coding: utf-8 -*-
@Time    :   2024/3/20 21:07
@Auther  :   zyh
@Email   :
@Project :   MultiQbvScheduler
@File    :   plot.py
'''
import os
import shutil
import matplotlib.pyplot as plt
import random

color_list = ['seagreen', 'skyblue', 'lime', 'maroon', 'lightsalmon',
              'grey', 'goldenrod', 'forestgreen', 'darkkhaki', 'coral',
              'aliceblue', 'blueviolet', 'blanchedalmond', 'cyan', 'darkgray',
              'deeppink']


def generate_random_colors_hex(x_num_colors, y_num_colors):
    colors = []
    for x in range(x_num_colors):
        colors.append(list())
        for y in range(y_num_colors):
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)
            color_hex = "#{:02x}{:02x}{:02x}".format(red, green, blue)
            colors[x].append(color_hex)
    return colors


def visualize_timeline(timelines: dict, hyper_period):
    print("============== visualize timeline ==============")
    if "output" in os.listdir("./"):
        shutil.rmtree("./output/")
    os.mkdir("./output/")
    for name in timelines:
        draw_gantt_chart(name, timelines[name], hyper_period)
    return 0


def draw_gantt_chart(name: str, timeline: list[list[int]], hyper_period):
    plt.figure(figsize=(12, 4))
    colors = generate_random_colors_hex(10, 100)

    for i in range(len(timeline)):
        if 'node' in name:
            start_time, end_time, position, _ = timeline[i]
        else:
            start_time, end_time, position, cycle_if_stream, m_uni_id = timeline[i]
        duration = end_time - start_time
        if 'node' in name:
            # plt.barh(7, 5000, left=start_time - 5000, height=0.4, color='lime', edgecolor='black')
            plt.barh(position, duration, left=start_time, height=0.4, color='skyblue', edgecolor='black')
        else:
            plt.barh(position, duration, left=start_time, height=0.4, color=colors[cycle_if_stream][m_uni_id], edgecolor='black')

    plt.title(name)
    plt.xlabel('Time')
    if 'node' in name:
        y_label_name = 'Queue'
    else:
        y_label_name = 'Device'
    plt.ylabel(y_label_name)

    plt.xlim(0, hyper_period)

    positions = [item[2] for item in timeline]
    plt.yticks(positions, [f'{y_label_name} {p}' for p in positions])

    plt.grid(True)

    plt.tight_layout()
    if "savefig" in os.listdir("./"):
        shutil.rmtree("./savefig/")
    os.mkdir("./savefig/")
    plt.savefig(f"./savefig/{name}.jpg", dpi=300)
    plt.show()

    return 0

