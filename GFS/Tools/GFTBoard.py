"""
@Author: P_k_y
@Time: 2021/1/4
"""
import pickle
from dearpygui.core import *
from dearpygui.simple import *
import os
import json

from  GFS.FIS.RuleLib import RuleLib


def plot_callback():
    """
    绘制训练中的 Fitness 曲线图。
    @return: None
    """
    max_f = get_data("max_fitness")
    average_f = get_data("average_fitness")
    min_f = get_data("min_fitness")
    xs = [i for i in range(len(max_f))]

    add_line_series("Training Log for GFS Algorithm", "Max Fitness", xs, max_f, weight=2, color=[0, 200, 200])
    add_line_series("Training Log for GFS Algorithm", "Average Fitness", xs, average_f, weight=2, color=[200, 0, 200])
    add_line_series("Training Log for GFS Algorithm", "Min Fitness", xs, min_f, weight=2, color=[0, 200, 0])

    add_shade_series("Training Log for GFS Algorithm", "Max Fitness", xs, max_f, weight=2, fill=[0, 200, 200, 100])
    add_shade_series("Training Log for GFS Algorithm", "Average Fitness", xs, average_f, weight=2, fill=[200, 0, 200, 100])
    add_shade_series("Training Log for GFS Algorithm", "Min Fitness", xs, min_f, weight=2, fill=[0, 200, 0, 100])


def choose_log_file(sender, data):
    """
    选择 GFT Log 文件。
    @return: None
    """
    open_file_dialog(callback=apply_select_log_file, extensions='.gft,.*')


def choose_individual_file(sender, data):
    """
    选择 GFT Log 文件。
    @return: None
    """
    open_file_dialog(callback=apply_select_individual_file, extensions='.json,.*')


def view_gft_info(gft_log):
    """
    查看训练超参数，突变概率，交叉概率等。
    @param gft_log: log 字典
    @return: None
    """
    if not get_data("first_load_hyper"):
        delete_item("HyperParameters")

    with window("HyperParameters", width=200, height=200, no_move=True, no_resize=True):
        set_window_pos("HyperParameters", 1040, 60)
        add_text("cross pro: {:.2f}".format(gft_log["cross_pro"]))
        add_text("mutate pro: {:.2f}".format(gft_log["mutate_pro"]))
        add_text("population size: {}".format(gft_log["population_size"]))
        add_text("episode: {}".format(gft_log["episode"]))
        add_text("Total Time: {:.2f}s".format(sum(gft_log["fitness_history"]["time_used_list"])))
        add_text("parallelized: {}".format(gft_log["parallelized"]))

    add_data("first_load_hyper", False)


def apply_select_log_file(sender, data):
    """
    每次选择 Log 文件后需要执行的函数，重载训练参数。
    @return: None
    """
    directory = data[0]
    file = data[1]

    f = open(os.path.join(directory, file), 'rb')
    gft_log = pickle.load(f)

    add_data("max_fitness", gft_log["fitness_history"]["max_fitness_list"])
    add_data("average_fitness", gft_log["fitness_history"]["average_fitness_list"])
    add_data("min_fitness", gft_log["fitness_history"]["min_fitness_list"])
    add_data("rule_lib_list", gft_log["rule_lib_list"])

    plot_callback()
    view_gft_info(gft_log)


def visualize_individual(individual: dict):
    """
    将 Individual 可视化为人类方便理解的规则库。
    @param individual: 个体对象
    @return: None
    """
    rules_text_list = []

    for i, rule_lib in enumerate(get_data("rule_lib_list")):
        rules_text_list.append("\n== Rule Lib[No.%d] ==" % i)
        rule_lib.encode_by_chromosome(individual["rule_lib_chromosome"][i])
        rules = rule_lib.decode()
        for idx, rule in enumerate(rules):
            rules_text_list.append("[%3d]" % idx + rule.get_rule_str())

    if not get_data("first_load_individual"):
        delete_item("Visualize Individual")

    with window("Visualize Individual", width=800, height=400):
        set_window_pos("Visualize Individual", 400, 200)
        for text in rules_text_list:
            add_text(text)

    add_data("first_load_individual", False)


def apply_select_individual_file(sender, data):
    """
    每次选择 Individual 个体后需要执行的函数，可视化规则库。
    @return: None
    """
    directory = data[0]
    file = data[1]
    f = open(os.path.join(directory, file), 'r')
    individual = json.load(f)

    visualize_individual(individual)


if __name__ == '__main__':

    with window("TrainLogWindow", width=500, height=500):
        add_button("Choose Log File", callback=choose_log_file, tip="Select Log File, usually named 'events.out.gft'.")
        add_same_line(spacing=10)
        add_button("Choose Individual", callback=choose_individual_file, tip="Visualize the Individual with human's rule.")

        add_plot("Training Log for GFS Algorithm", height=-1)
        add_data("max_fitness", [0])
        add_data("average_fitness", [0])
        add_data("min_fitness", [0])
        add_data("rule_lib_list", [])
        add_data("first_load_hyper", True)
        add_data("first_load_individual", True)

    start_dearpygui(primary_window="TrainLogWindow")
