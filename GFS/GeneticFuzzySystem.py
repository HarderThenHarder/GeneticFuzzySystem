"""
@Author: P_k_y
@Time: 2020/12/14
"""
import json
import random
import multiprocessing
from .FIS.RuleLib import RuleLib
from .FIS.DecisionSystem import DecisionSystem
from .FIS.DecisionSystemSimulation import DecisionSystemSimulation
import os
import numpy as np
from abc import ABCMeta, abstractmethod
import copy
import time
import matplotlib.pyplot as plt


class BaseGFT(metaclass=ABCMeta):
    def __init__(self, rule_lib_list, population_size, episode, mutation_pro, cross_pro, simulator, parallelized=False):
        """
        GFT基类。
        @param rule_lib_list: 规则库对象
        @param population_size: 种群规模（存在的染色体条数，可以理解为存在的规则库个数）
        @param episode: 训练多少轮
        @param mutation_pro: 变异概率
        @param cross_pro: 交叉概率
        @param simulator: 仿真环境对象，用于获取观测、回报等
        """
        self.rule_lib_list = rule_lib_list
        self.population_size = population_size
        self.episode = episode
        self.mutation_pro = mutation_pro
        self.cross_pro = cross_pro
        self.population = []
        self.simulator = simulator
        self.parallelized = parallelized
        self.fitness_history = {"min_fitness_list": [],
                                "max_fitness_list": [],
                                "average_fitness_list": []}

    def init_population(self) -> None:
        """
        种群初始化函数，初始化规定数目个个体（Individual），一个个体的染色体为二维数组，
        其中每一维代表一个特定规则库的染色体。
        @return: None
        """
        for i in range(self.population_size):
            rule_lib_chromosome = []
            mf_chromosome = []

            fis_num = len(self.rule_lib_list)
            """ 由于存在多个规则库，不同规则库由不同的FIS决策器决策不同行为，因此需要为每一个规则库生成一条随机染色体 """
            for index in range(fis_num):
                output_fuzzy_variable = self.rule_lib_list[index].fuzzy_variable_list[-1]
                output_terms = output_fuzzy_variable.all_term()

                """ 将输出模糊变量的term按id数字化 """
                genes = [term.id for term in output_terms]
                all_term_list = [term for fuzzy_variable in self.rule_lib_list[index].fuzzy_variable_list
                                 for term in fuzzy_variable.all_term()]
                current_rule_lib_chromosome_size = len(self.rule_lib_list[index].rule_lib)

                """ 默认使用三角隶属函数，因此隶属函数染色体长度等于隶属函数个数*3 """
                current_mf_chromosome_size = len(all_term_list) * 3
                current_rule_lib_chromosome = [genes[random.randint(0, len(genes) - 1)] for _ in
                                               range(current_rule_lib_chromosome_size)]
                current_mf_chromosome = [random.randint(-10, 10) for _ in range(current_mf_chromosome_size)]

                """ 往个体的染色体中加入代表当前规则库的染色体 """
                rule_lib_chromosome.append(current_rule_lib_chromosome)
                mf_chromosome.append(current_mf_chromosome)

            individual = {"rule_lib_chromosome": rule_lib_chromosome, "mf_chromosome": mf_chromosome, "fitness": 0,
                          "flag": 0}

            self.population.append(individual)

    def compute_fitness(self, individual: dict, simulator, individual_id=None, queue=None, min_v=None, max_v=None, average_num=1) -> float:
        """
        计算个体的适应值，需要先将个体的染色体解析成模糊规则库，构成GFT中的FIS，依据该决策器进行仿真，根据最终的仿真结果进行适应值计算，计算方式可以自定义。
        @param individual_id: 个体id，用于多进程计算的时候确定该个体
        @param queue: 进程队列，当该进程完成计算后往队列中添加一个信号，便于主进程统计
        @param simulator: simulator 对象，用于并行计算
        @param average_num: 取N次实验的实验结果作为返回值
        @param max_v: 若要对reward进行clip，则设置该最大值，默认为None，不做clip
        @param min_v: 若要对reward进行clip，则设置该最小值，默认为None，不做clip
        @param individual: 个体单位
        @return: 该个体的适应值
        """

        if self.parallelized:
            assert queue, "Parallelized Mode Need multiprocessing.Queue() object, please pass @param: queue."
            assert individual_id is not None, "Parallelized Mode Need individual ID, please pass @param: individual_id."

        gft_controllers = []
        for index, rule_lib in enumerate(self.rule_lib_list):
            """ 将RuleLib中的数字编码染色体解析为包含多个Rule对象的列表 """
            rl = RuleLib(rule_lib.fuzzy_variable_list)
            rl.encode_by_chromosome(individual["rule_lib_chromosome"][index])
            rules = rl.decode()

            """ 对模糊变量的隶属函数参数进行重计算，根据染色体中的隶属函数偏移offset来更改隶属函数三角形的三个点坐标 """
            new_fuzzy_variable_list = copy.deepcopy(rule_lib.fuzzy_variable_list)
            count = 0
            for fuzzy_variable in new_fuzzy_variable_list:
                for k, v in fuzzy_variable.terms.items():
                    if (v.trimf[0] == v.trimf[1] and v.trimf[1] == v.trimf[2] and v.trimf[2] == 0) or (
                            v.trimf[0] == -666):  # 类别型模糊变量
                        count += 3
                        continue
                    offset_value = v.span / 20
                    new_a = v.trimf[0] + individual["mf_chromosome"][index][count] * offset_value
                    new_b = v.trimf[1]
                    new_c = v.trimf[2] + individual["mf_chromosome"][index][count + 2] * offset_value
                    new_tri = [new_a, new_b, new_c]
                    new_tri.sort()  # 因为平移有正有负，平移后可能会出现a点平移到了b点右边，但三角形的三个点不能乱序，因此需要按从小到大排序
                    count += 3
                    v.trimf = new_tri

            """ 构建FIS推理器 """
            now_ds = DecisionSystem(new_fuzzy_variable_list, rules)
            now_dss = DecisionSystemSimulation(now_ds)
            gft_controllers.append(now_dss)

        sum_score = 0
        """ 取N次实验平均值返回结果 """
        for i in range(average_num):
            current_reward = self.start_simulation(gft_controllers, simulator)
            current_reward = min(min_v, current_reward) if min_v else current_reward  # 单局仿真最小值clip
            current_reward = max(max_v, current_reward) if max_v else current_reward  # 单局仿真最大值clip
            sum_score += current_reward
        sum_score /= average_num

        if self.parallelized:
            queue.put((individual_id, sum_score))
        else:
            """ 非并行可以直接在该函数内为个体赋值，若使用并行进程则需要通过返回值在主进程中完成个体赋值 """
            individual["fitness"] = sum_score
            individual["flag"] = 1

        return sum_score

    def visualize_progress(self, epoch: int, total_epoch: int, step: int, total_step: int) -> None:
        """
        可视化当前计算的进度条。
        @param epoch: 当前epoch轮数
        @param total_epoch: 总共epoch轮数
        @param step: 当前步数
        @param total_step: 总步数
        @return: None
        """
        max_len = 40
        current_progress = int(step / total_step * 40)
        print('\r[Epoch: %d/%d][' % (epoch, total_epoch) + '=' * current_progress + '-' * (
                max_len - current_progress) + ']', end='')

    def save_train_history(self, save_log_path="models/train_log.png"):
        """
        保存训练曲线到本地文件中。
        @param save_log_path: 曲线图存储路径
        @return: None
        """
        plt.clf()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title("Training Log For GFS Algorithm")
        plt.xlabel("Epoch(s)")
        plt.ylabel("Fitness")
        plt.plot(self.fitness_history["min_fitness_list"], color='green', alpha=0.5, label='Min Fitness', linestyle='-.')
        plt.plot(self.fitness_history["average_fitness_list"], color='r', alpha=0.8, label='Average Fitness')
        plt.plot(self.fitness_history["max_fitness_list"], color='c', alpha=0.5, label='Max Fitness', linestyle='-.')
        plt.legend()
        plt.savefig(save_log_path)

    def compute_with_parallel(self, epoch, total_epoch):
        """
        并行计算每一个个体的适应值Fitness。
        @return: None
        """
        """ 统计 flag == 0（未被计算过适应值）的个体对象 """
        uncalculated_individual_list = list(filter(lambda x: x["flag"] == 0, self.population))
        sum_count, current_count = len(uncalculated_individual_list), 0
        simulator_list = [copy.deepcopy(self.simulator) for _ in range(len(uncalculated_individual_list))]
        process_list, q = [], multiprocessing.Queue()

        """ 为每一个待计算的染色体建立一个新进程 """
        for count, individual in enumerate(uncalculated_individual_list):
            process = multiprocessing.Process(target=self.compute_fitness, args=(individual, simulator_list[count], count, q))
            process.start()
            process_list.append(process)

        """ 可视化进度条，当所有染色体进程完成计算后退出该函数 """
        self.visualize_progress(epoch, total_epoch, current_count, sum_count)
        while True:
            individual_id, fitness = q.get()
            uncalculated_individual_list[individual_id]["fitness"] = fitness
            uncalculated_individual_list[individual_id]["flag"] = 1
            current_count += 1
            self.visualize_progress(epoch, total_epoch, current_count, sum_count)
            if current_count == sum_count:
                break

    def compute_without_parallel(self, epoch, total_epoch):
        """
        串行计算每一个个体的适应值Fitness。
        @return: None
        """
        sum_count = len(self.population)
        for count, individual in enumerate(self.population):
            count += 1
            self.visualize_progress(epoch, total_epoch, count, sum_count)
            if individual["flag"] == 0:
                self.compute_fitness(individual, self.simulator)

    def select(self, epoch: int, total_epoch: int) -> None:
        """
        根据fitness来计算每一条染色体被选择的概率，按照概率来选择染色体是否被保留。
        @param epoch: 当前执行的轮数
        @param total_epoch: 总共须迭代的轮数
        @return: None
        """
        start = time.time()

        if self.parallelized:
            self.compute_with_parallel(epoch, total_epoch)
        else:
            self.compute_without_parallel(epoch, total_epoch)

        self.population = sorted(self.population, key=lambda x: x["fitness"])
        fitness_list = [x["fitness"] for x in self.population]
        fitness_list_for_choice = copy.deepcopy(fitness_list)

        if min(fitness_list_for_choice) < 0:
            """ 如果列表中有负数，则将列表整体平移使其列表中全为正数，保证后面计算概率正确, 加1e-6是为了保证fitness全为正，计算的概率不为0 """
            fitness_list_for_choice = [x - min(fitness_list_for_choice) + 1e-6 for x in fitness_list_for_choice]

        sum_fitness = sum(fitness_list_for_choice)
        fit_pro = [fitness / sum_fitness for fitness in fitness_list_for_choice]

        """ 按照概率分布选择出种群规模条染色体 """
        selected_population = np.random.choice(self.population, self.population_size, replace=False, p=fit_pro)

        use_time = time.time() - start
        max_f, average_f, min_f = max(fitness_list), sum(fitness_list) / len(fitness_list), min(fitness_list)
        self.fitness_history["min_fitness_list"].append(min_f)
        self.fitness_history["average_fitness_list"].append(average_f)
        self.fitness_history["max_fitness_list"].append(max_f)

        self.save_train_history()

        print("  Min Fitness: %.2f  |  Max Fitness: %.2f  |  Average Fitness: %.2f  |  Time Used: %.1fs" % (
            min_f, max_f, average_f, use_time))
        self.population = list(selected_population)

    def get_offspring(self, parent: list) -> list:
        """
        根据两个父代个体（Individual）进行交叉后，返回两条新的子代个体（Individual）。
        @param parent: 父代 individual list
        @return: 子代 individual list
        """
        offspring = copy.deepcopy(parent)

        """ 对规则库列表中的每一个规则库进行交叉互换，但只能不同染色体上的相同规则库之间才能进行交叉，通过index来确保相同类型的规则库 """
        for index, rule_lib in enumerate(self.rule_lib_list):
            all_term_list = [copy.deepcopy(fuzzy_variable.all_term()) for fuzzy_variable in
                             rule_lib.fuzzy_variable_list]
            current_rule_lib_chromosome_size = len(rule_lib.rule_lib)
            current_mf_chromosome_size = len(all_term_list) * 3

            """ 随机选择交换基因段的左右位点索引（规则库染色体） """
            cross_left_position_rule_lib = random.randint(0, current_rule_lib_chromosome_size - 1)
            cross_right_position_rule_lib = random.randint(cross_left_position_rule_lib,
                                                           current_rule_lib_chromosome_size - 1)

            """ 交换子代对应位置的基因片段 """
            offspring[0]["rule_lib_chromosome"][index][cross_left_position_rule_lib:cross_right_position_rule_lib + 1], \
            offspring[1]["rule_lib_chromosome"][index][cross_left_position_rule_lib:cross_right_position_rule_lib + 1] = \
                offspring[1]["rule_lib_chromosome"][index][
                cross_left_position_rule_lib:cross_right_position_rule_lib + 1], \
                offspring[0]["rule_lib_chromosome"][index][
                cross_left_position_rule_lib:cross_right_position_rule_lib + 1]

            """ 随机选择交换基因段的左右位点索引（隶属函数染色体） """
            cross_left_position_mf = random.randint(0, current_mf_chromosome_size - 1)
            cross_right_position_mf = random.randint(cross_left_position_mf, current_mf_chromosome_size - 1)

            offspring[0]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1], \
            offspring[1]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1] = \
                offspring[1]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1], \
                offspring[0]["mf_chromosome"][index][cross_left_position_mf:cross_right_position_mf + 1]

        """ 新的子代没有被Simulation过，flag置为0，代表需要通过simulation后来计算适应值fitness """
        offspring[0]["flag"] = 0
        offspring[1]["flag"] = 0
        return offspring

    def cross(self) -> None:
        """
        将一个种群（population）中的所有个体（Individual）按概率进行交叉互换，
        并将子代添加入当前种群中。
        @return: None
        """
        offspring = []
        random.shuffle(self.population)
        """ 相邻两个个体之间进行交叉 """
        d = list(range(0, len(self.population), 2))
        for i in d:
            pro = random.random()
            if pro < self.cross_pro:
                now_offspring = self.get_offspring(self.population[i: i + 2])
                offspring.extend(now_offspring)
        self.population.extend(offspring)

    def mutate(self) -> None:
        """
        基因变异函数，对种群中每一个个体（Individual），随机选择染色体中的某一段，
        对该段中的每一个基因执行一次基因突变。
        @return: None
        """
        for individual in self.population:
            pro = random.random()

            """ 对每一个规则库都要进行突变 """
            if pro < self.mutation_pro:
                for index, rule_lib in enumerate(self.rule_lib_list):
                    output_fuzzy_variable = rule_lib.fuzzy_variable_list[-1]
                    output_terms = output_fuzzy_variable.all_term()
                    genes = [term.id for term in output_terms]
                    all_term_list = [term for fuzzy_variable in self.rule_lib_list[index].fuzzy_variable_list
                                     for term in fuzzy_variable.all_term()]

                    current_rule_lib_chromosome_size = len(rule_lib.rule_lib)
                    current_mf_chromosome_size = len(all_term_list) * 3

                    """ 选取突变点，并将该点之后的全部基因段进行基因突变（规则库） """
                    mutation_pos_rule_lib = random.randint(0, current_rule_lib_chromosome_size - 1)
                    gene_num = len(genes)
                    individual["rule_lib_chromosome"][index][mutation_pos_rule_lib:] = [random.randint(0, gene_num - 1)
                                                                                        for _ in
                                                                                        range(
                                                                                            current_rule_lib_chromosome_size -
                                                                                            mutation_pos_rule_lib)]

                    """ 选取突变点，并将该点之后的全部基因段进行基因突变（隶属函数） """
                    mutation_pos_mf = random.randint(0, current_mf_chromosome_size - 1)
                    individual["mf_chromosome"][index][mutation_pos_mf:] = [random.randint(-10, 10) for _ in
                                                                            range(
                                                                                current_mf_chromosome_size - mutation_pos_mf)]

                """ 变异后的个体flag需被重置 """
                individual["flag"] = 0

    def get_optimal_individual(self):
        """
        获取 fitness 最高的个体对象。
        @return: 最优 Individual
        """
        sorted_population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        return sorted_population[0]

    def save_all_population(self, file_path: str):
        """
        将整个种群存放入文件当中（用于保存模型checkpoint，便于下次接着本次训练）。
        @param file_path: 文件保存目录
        @return: None
        """
        with open(file_path, "w") as f:
            all_population_dict = {"all_population": self.population}
            json.dump(all_population_dict, f)

    def load_all_population(self, file_path):
        """
        从文件中载入整个种群。
        @param file_path: 文件保存目录。
        @return: None
        """
        try:
            with open(file_path, "r") as f:
                all_population_dict = json.load(f)
                self.population = all_population_dict["all_population"]
        except:
            raise IOError("[ERROR] Open File Failed!")

    def save_optimal_individual_to_file(self, path_rule_lib, path_mf, path_individual, optimal_individual):
        """
        将得分值最高的个体（Individual）存入文件中。
        @param path_individual: 最优个体存放目录（不包含文件后缀名）
        @param path_rule_lib: 规则库文件存放目录（不包含文件后缀名）
        @param path_mf: 隶属函数参数文件存放目录（不包含文件后缀名）
        @param optimal_individual: 最优个体对象
        @return: None
        """
        for index, rule_lib in enumerate(self.rule_lib_list):
            """ 将每一个规则库存分别放入不同的文件中，文件名中包含规则库编号，如：RuleLib[No.1]，代表1号规则库 """
            current_path_rule_lib = path_rule_lib + "_[No." + str(index) + "].json"
            current_path_mf = path_mf + "_[No." + str(index) + "].json"

            rule_lib.encode_by_chromosome(optimal_individual["rule_lib_chromosome"][index])

            """ 将规则库和隶属函数和个体对象均保存到本地 """
            rule_lib.save_rule_base_to_file(current_path_rule_lib)
            rule_lib.save_mf_to_file(current_path_mf, optimal_individual)

            """ 最优个体对象，只用存储一次 """
            if not index:
                current_path_individual = path_individual + ".json"
                rule_lib.save_individual_to_file(current_path_individual, optimal_individual)

    def train(self, save_best_rulelib_mf_path="RuleLibAndMF", save_all_path="AllPopulations",
              save_best_individual_path="OptimalIndividuals", base_path="models") -> None:
        """
        遗传算法训练函数。
        @param base_path: 模型存放总路径
        @param save_all_path: 种群存放路径
        @param save_best_rulelib_mf_path: 最优个体存放路径
        @param save_best_individual_path: 个体模型存放路径
        @return: None
        """

        """ 若目录不存在则新建目录文件夹 """
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        save_best_rulelib_mf_path = os.path.join(base_path, save_best_rulelib_mf_path)
        save_all_path = os.path.join(base_path, save_all_path)
        save_best_individual_path = os.path.join(base_path, save_best_individual_path)

        if not os.path.exists(save_best_rulelib_mf_path):
            os.mkdir(save_best_rulelib_mf_path)
        if not os.path.exists(save_best_individual_path):
            os.mkdir(save_best_individual_path)
        if not os.path.exists(save_all_path):
            os.mkdir(save_all_path)

        print("\nStart to Initialize Rule Lib...")
        self.init_population()
        print("\nFinished Initialized Rule Lib, Start to train...\n")
        for count in range(self.episode):
            count += 1
            self.cross()
            self.mutate()
            self.select(count, self.episode)
            self.save_all_population(os.path.join(save_all_path, "all_population{}.json".format(count)))
            optimal_individual = self.get_optimal_individual()
            self.save_optimal_individual_to_file(
                os.path.join(save_best_rulelib_mf_path,
                             "[Epoch_{}]RuleLib({:.1f})".format(count, optimal_individual["fitness"])),
                os.path.join(save_best_rulelib_mf_path,
                             "[Epoch_{}]MF({:.1f})".format(count, optimal_individual["fitness"])),
                os.path.join(save_best_individual_path,
                             "[Epoch_{}]Individual({:.1f})".format(count, optimal_individual["fitness"])),
                optimal_individual)

    def evaluate(self, model_name: str):
        """
        使用已存储模型，查看训练效果。
        @return: None
        """
        individual = json.load(open(model_name, 'r'))

        print("\nLoading Model...")

        gft_controllers = []
        for index, rule_lib in enumerate(self.rule_lib_list):
            """ 将RuleLib中的数字编码染色体解析为包含多个Rule对象的列表 """
            rl = RuleLib(rule_lib.fuzzy_variable_list)
            rl.encode_by_chromosome(individual["rule_lib_chromosome"][index])
            rules = rl.decode()

            """ 对模糊变量的隶属函数参数进行重计算，根据染色体中的隶属函数偏移offset来更改隶属函数三角形的三个点坐标 """
            new_fuzzy_variable_list = copy.deepcopy(rule_lib.fuzzy_variable_list)
            count = 0
            for fuzzy_variable in new_fuzzy_variable_list:
                for k, v in fuzzy_variable.terms.items():
                    if (v.trimf[0] == v.trimf[1] and v.trimf[1] == v.trimf[2] and v.trimf[2] == 0) or (
                            v.trimf[0] == -666):  # 类别型模糊变量
                        count += 3
                        continue
                    offset_value = v.span / 20
                    new_a = v.trimf[0] + individual["mf_chromosome"][index][count] * offset_value
                    new_b = v.trimf[1]
                    new_c = v.trimf[2] + individual["mf_chromosome"][index][count + 2] * offset_value
                    new_tri = [new_a, new_b, new_c]
                    new_tri.sort()  # 因为平移有正有负，平移后可能会出现a点平移到了b点右边，但三角形的三个点不能乱序，因此需要按从小到大排序
                    count += 3
                    v.trimf = new_tri

            """ 构建FIS推理器 """
            now_ds = DecisionSystem(new_fuzzy_variable_list, rules)
            now_dss = DecisionSystemSimulation(now_ds)
            gft_controllers.append(now_dss)

        print("\nStart Simulation...")

        self.start_simulation(gft_controllers, self.simulator)

    @abstractmethod
    def start_simulation(self, controllers: list, simulator) -> float:
        """
        仿真器，用于根据FIS决策器的行为决策更新，获取并返回fitness。
        @param simulator: 仿真器对象。
        @param controllers: 包含继承自DecisionSystemSimulation的仿真器对象列表
        @return: 适应值
        """
        pass


if __name__ == '__main__':
    print("Hello GFS!")
