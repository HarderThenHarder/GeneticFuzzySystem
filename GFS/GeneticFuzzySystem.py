"""
@Author: P_k_y
@Time: 2020/12/14
"""
import json
import random
import copy
from GFS.FIS.RuleLib import RuleLib
from GFS.FIS.DecisionSystem import DecisionSystem
from GFS.FIS.DecisionSystemSimulation import DecisionSystemSimulation
import os
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseGFS(metaclass=ABCMeta):
    def __init__(self, rule_lib, population_size, episode, mutation_pro, cross_pro):
        """
        BaseGFS基类。
        @param rule_lib: 规则库对象
        @param population_size: 种群规模（存在的染色体条数，可以理解为存在的规则库个数）
        @param episode: 训练多少轮
        @param mutation_pro: 变异概率
        @param cross_pro: 交叉概率
        """
        self.rule_lib = rule_lib
        self.population_size = population_size
        self.rulebase_chromosome_size = len(self.rule_lib.rule_lib)
        self.all_term_list = []
        for fuzzy_variable in self.rule_lib.fuzzy_variable_list:
            fuzzy_variable_terms = fuzzy_variable.all_term()
            self.all_term_list.extend(copy.deepcopy(fuzzy_variable_terms))
        self.mf_chromosome_size = len(self.all_term_list) * 3
        self.episode = episode
        self.mutation_pro = mutation_pro
        self.cross_pro = cross_pro
        self.population = []
        self.genes = []
        output_fuzzy_variable = self.rule_lib.fuzzy_variable_list[-1]
        output_terms = output_fuzzy_variable.all_term()
        for term in output_terms:
            self.genes.append(term.id)

    def init_population(self) -> None:
        """
        种群初始化函数，用于初始化规定数目个个体（Individual）。
        @return: None
        """
        for i in range(0, self.population_size):
            """ 初始化一条随机染色体，从输出模糊变量中随机选择N个模糊程度（Term） """
            rulebase_chromosome = [self.genes[random.randint(0, len(self.genes) - 1)] for _ in
                                   range(self.rulebase_chromosome_size)]

            """ 初始化隶属函数偏移的一条随机染色体，默认隶属函数偏移值在[-10, 10] """
            mf_chromosome = [random.randint(-10, 10) for _ in range(self.mf_chromosome_size)]

            """ 一个个体（Individual）包括规则染色体（rulebase_chromosome）和隶属函数偏移染色体（mf_chromosome），flag代表该染色体是否被Simulation过 """
            individual = {"rulebase_chromosome": rulebase_chromosome, "mf_chromosome": mf_chromosome, "fitness": 0,
                          "flag": 0}

            self.population.append(individual)

    def compute_fitness(self, individual: dict) -> float:
        """
        计算个体的适应值，需要先将个体的染色体解析成模糊规则库，构成FIS决策器，依据该决策器进行仿真，根据最终的仿真结果进行适应值计算，计算方式可以自定义。
        @param individual: 个体单位
        @return: 该个体的适应值
        """
        rb = RuleLib(self.rule_lib.fuzzy_variable_list)
        rb.encode_by_chromosome(individual["rulebase_chromosome"])
        rules = rb.decode()
        new_fuzzy_variable_list = copy.deepcopy(self.rule_lib.fuzzy_variable_list)
        count = 0
        for fuzzy_variable in new_fuzzy_variable_list:
            for k, v in fuzzy_variable.terms.items():
                if v.trimf[0] == v.trimf[1] and v.trimf[1] == v.trimf[2] and v.trimf[2] == 0:  # 类别型模糊变量
                    continue
                offset_value = v.span / 10
                new_a = v.trimf[0] + individual["mf_chromosome"][count] * offset_value
                new_b = v.trimf[1] + individual["mf_chromosome"][count + 1] * offset_value
                new_c = v.trimf[2] + individual["mf_chromosome"][count + 2] * offset_value
                new_tri = [new_a, new_b, new_c]
                new_tri.sort()
                count += 3
                v.trimf = new_tri

        ds = DecisionSystem(new_fuzzy_variable_list, rules)
        dss = DecisionSystemSimulation(ds)
        sum_socre = 0
        count = 0
        average_num = 3

        """ 取三次实验的平均值作为最终结果 """
        for i in range(average_num):
            count += 1
            sum_socre += self.start_simulation(dss)
        return sum_socre / average_num

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
        print('\r[Epoch: %d/%d][' % (epoch, total_epoch) + '=' * current_progress + '-' * (max_len - current_progress) + ']', end='')

    def select(self, epoch: int, total_epoch: int) -> None:
        """
        根据fitness来计算每一条染色体被选择的概率，按照概率来选择染色体是否被保留。
        @param epoch: 当前执行的轮数
        @param total_epoch: 总共须迭代的轮数
        @return: None
        """
        sum_count = len(self.population)
        for count, individual in enumerate(self.population):
            count += 1
            self.visualize_progress(epoch, total_epoch, count, sum_count)
            if individual["flag"] == 0:
                individual["fitness"] = self.compute_fitness(individual)
                individual["flag"] = 1

        self.population = sorted(self.population, key=lambda x: x["fitness"])
        fitness_list = [x["fitness"] for x in self.population]
        sum_fitness = sum(fitness_list)
        fit_pro = [fitness / sum_fitness for fitness in fitness_list]

        """ 按照概率分布选择出种群规模条染色体 """
        selected_population = np.random.choice(self.population, self.population_size, replace=False, p=fit_pro)
        print("  Min Fitness: %.2f  |  Max Fitness: %.2f  |  Average Fitness: %.2f" % (min(fitness_list), max(fitness_list), sum_fitness / len(fitness_list)))
        self.population = list(selected_population)

    def select_by_fitness(self) -> None:
        """
        优胜劣汰算法，直接选择适应值较大的染色体保留，该方法是简化版的选择方法，建议使用select()函数。
        @return: None
        """
        for count, individual in enumerate(self.population):
            count += 1
            if individual["flag"] == 0:
                individual["fitness"] = self.compute_fitness(individual)
                individual["flag"] = 1
        self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        self.population = self.population[:self.population_size]

    def get_offspring(self, parent: list) -> list:
        """
        根据两个父代个体（Individual）进行交叉后，返回两条新的子代个体（Individual）。
        @param parent: 父代 individual list
        @return: 子代 individual list
        """
        offspring = copy.deepcopy(parent)

        """ 随机选择交换基因段的左右位点索引（规则库染色体） """
        cross_left_position_rulebase = random.randint(0, self.rulebase_chromosome_size - 1)
        cross_right_position_rulebase = random.randint(cross_left_position_rulebase, self.rulebase_chromosome_size - 1)

        """ 交换子代对应位置的基因片段 """
        offspring[0]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1], \
            offspring[1]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1] = \
            offspring[1]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1], \
            offspring[0]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1]

        """ 随机选择交换基因段的左右位点索引（隶属函数染色体） """
        cross_left_position_mf = random.randint(0, self.mf_chromosome_size - 1)
        cross_right_position_mf = random.randint(cross_left_position_mf, self.mf_chromosome_size - 1)

        offspring[0]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1], \
            offspring[1]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1] = \
            offspring[1]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1], \
            offspring[0]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1]

        """ 新的子代没有被Simulation过，flag置为0，代表需要通过simulation后来计算适应值fitness """
        offspring[0]["flag"] = 0
        offspring[1]["flag"] = 0
        return offspring

    def cross(self) -> None:
        """
        将一个种群（population）中的所有个体（Individual）按概率进行交叉互换，并将子代添加入当前种群中。
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
        基因变异函数，对种群中每一个个体（Individual），随机选择染色体中的某一段，对该段中的每一个基因执行一次基因突变。
        @return: None
        """
        for individual in self.population:
            pro = random.random()
            if pro < self.mutation_pro:
                """ 选取突变点，并将该点之后的全部基因段进行基因突变（规则库） """
                mutation_pos_rulebase = random.randint(0, self.rulebase_chromosome_size - 1)
                gene_num = len(self.genes)
                individual["rulebase_chromosome"][mutation_pos_rulebase:] = [random.randint(0, gene_num - 1) for _ in
                                                                             range(
                                                                                 self.rulebase_chromosome_size - mutation_pos_rulebase)]

                """ 选取突变点，并将该点之后的全部基因段进行基因突变（隶属函数） """
                mutation_pos_mf = random.randint(0, self.mf_chromosome_size - 1)
                individual["mf_chromosome"][mutation_pos_mf:] = [random.randint(-10, 10) for _ in
                                                                 range(self.mf_chromosome_size - mutation_pos_mf)]
                """ 变异后的个体flag需被重置 """
                individual["flag"] = 0

    def get_optimal_individual(self):
        optimal_individual = self.population[0]
        for i in range(1, len(self.population) - 1):
            if self.population[i]['fitness'] > optimal_individual["fitness"]:
                optimal_individual = self.population[i]
        return optimal_individual

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

    def save_optimal_individual_to_file(self, path_rulebase, path_mf, optimal_individual):
        """
        将得分值最高的个体（Individual）存入文件中。
        @param path_rulebase: 规则库文件存放目录
        @param path_mf: 隶属函数参数文件存放目录
        @param optimal_individual: 最优个体对象
        @return: None
        """
        self.rule_lib.encode_by_chromosome(optimal_individual["rulebase_chromosome"])
        self.rule_lib.save_rule_base_to_file(path_rulebase)
        with open(path_mf, "w") as f:
            mf_dict = {"mf_offset": optimal_individual["mf_chromosome"]}
            json.dump(mf_dict, f)

    def train(self, save_best_individual_path="TrainedFile", save_all_path="SavedAllPopulation", base_path="models") -> None:
        """
        遗传算法训练函数。
        @param base_path: 模型存放总路径
        @param save_all_path: 种群存放路径
        @param save_best_individual_path: 最优个体存放路径
        @return: None
        """
        """ 若目录不存在则新建目录文件夹 """
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        save_best_individual_path = os.path.join(base_path, save_best_individual_path)
        save_all_path = os.path.join(base_path, save_all_path)
        if not os.path.exists(save_best_individual_path):
            os.mkdir(save_best_individual_path)
        if not os.path.exists(save_all_path):
            os.mkdir(save_all_path)

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
                os.path.join(save_best_individual_path, "RuleLib{}({}).json".format(count, optimal_individual["fitness"])),
                os.path.join(save_best_individual_path, "MF{}({}).json".format(count, optimal_individual["fitness"])),
                optimal_individual)

    @abstractmethod
    def start_simulation(self, simulator: DecisionSystemSimulation) -> float:
        """
        仿真器，用于根据FIS决策器的行为决策更新，获取并返回fitness。
        @param simulator: 继承自DecisionSystemSimulation的仿真器对象
        @return: 适应值
        """
        pass


class BaseGFT(metaclass=ABCMeta):
    def __init__(self, rule_lib_list, population_size, episode, mutation_pro, cross_pro):
        """
        GFT基类。
        @param rule_lib_list: 规则库对象
        @param population_size: 种群规模（存在的染色体条数，可以理解为存在的规则库个数）
        @param episode: 训练多少轮
        @param mutation_pro: 变异概率
        @param cross_pro: 交叉概率
        """
        self.rule_lib_list = rule_lib_list
        self.population_size = population_size
        self.episode = episode
        self.mutation_pro = mutation_pro
        self.cross_pro = cross_pro
        self.population = []

    def init_population(self) -> None:
        """
        种群初始化函数，初始化规定数目个个体（Individual），并为GFT下的每一个。
        @return: None
        """
        for i in range(self.population_size):
            rulebase_chromosome = []
            mf_chromosome = []

            """ 由于存在多个规则库，不同规则库由不同的FIS决策器决策不同行为，因此需要为每一个规则库生成 """
            fis_num = len(self.rule_lib_list)
            for index in range(fis_num):
                genes = []
                rule_len = len(self.RB_gft[index].fuzzy_variable_list)
                output_fuzzy_variable = self.RB_gft[index].fuzzy_variable_list[rule_len - 1]
                output_terms = output_fuzzy_variable.all_term()
                for term in output_terms:
                    genes.append(term.id)
                all_term_list = []
                for fuzzy_variable in self.RB_gft[index].fuzzy_variable_list:
                    fuzzy_variable_terms = fuzzy_variable.all_term()
                    all_term_list.extend(copy.deepcopy(fuzzy_variable_terms))
                now_rulebase_chromosome_size = len(self.RB_gft[index].rule_base)
                now_mf_chromosome_size = len(all_term_list) * 3
                now_rulebase_chromosome = []
                now_mf_chromosome = []
                for j in range(now_rulebase_chromosome_size):
                    rand_genes = random.randint(0, len(genes) - 1)
                    now_rulebase_chromosome.append(genes[rand_genes])
                for j in range(0, now_mf_chromosome_size):
                    rand_offset = random.randint(0, 20) - 10
                    now_mf_chromosome.append(rand_offset)
                rulebase_chromosome.append(now_rulebase_chromosome)
                mf_chromosome.append(now_mf_chromosome)
            individual = {"rulebase_chromosome": rulebase_chromosome, "mf_chromosome": mf_chromosome, "fitness": 0,
                          "flag": 0, "average_fitness": []}
            self.population.append(individual)

    def compute_fitness(self, individual: dict) -> float:
        """
        计算个体的适应值，需要先将个体的染色体解析成模糊规则库，构成FIS决策器，依据该决策器进行仿真，根据最终的仿真结果进行适应值计算，计算方式可以自定义。
        @param individual: 个体单位
        @return: 该个体的适应值
        """
        rb = RuleLib(self.RB.fuzzy_variable_list)
        rb.encode_by_chromosome(individual["rulebase_chromosome"])
        rules = rb.decode()
        new_fuzzy_variable_list = copy.deepcopy(self.RB.fuzzy_variable_list)
        count = 0
        for fuzzy_variable in new_fuzzy_variable_list:
            for k, v in fuzzy_variable.terms.items():
                if v.trimf[0] == v.trimf[1] and v.trimf[1] == v.trimf[2] and v.trimf[2] == 0:  # 类别型模糊变量
                    continue
                offset_value = v.span / 10
                new_a = v.trimf[0] + individual["mf_chromosome"][count] * offset_value
                new_b = v.trimf[1] + individual["mf_chromosome"][count + 1] * offset_value
                new_c = v.trimf[2] + individual["mf_chromosome"][count + 2] * offset_value
                new_tri = [new_a, new_b, new_c]
                new_tri.sort()
                count += 3
                v.trimf = new_tri

        ds = DecisionSystem(new_fuzzy_variable_list, rules)
        dss = DecisionSystemSimulation(ds)
        sum_socre = 0
        count = 0
        average_num = 3

        """ 取三次实验的平均值作为最终结果 """
        for i in range(average_num):
            count += 1
            sum_socre += self.start_simulation(dss)
        return sum_socre / average_num

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
        print('\r[Epoch: %d/%d][' % (epoch, total_epoch) + '=' * current_progress + '-' * (max_len - current_progress) + ']', end='')

    def select(self, epoch: int, total_epoch: int) -> None:
        """
        根据fitness来计算每一条染色体被选择的概率，按照概率来选择染色体是否被保留。
        @param epoch: 当前执行的轮数
        @param total_epoch: 总共须迭代的轮数
        @return: None
        """
        sum_count = len(self.population)
        for count, individual in enumerate(self.population):
            count += 1
            self.visualize_progress(epoch, total_epoch, count, sum_count)
            if individual["flag"] == 0:
                individual["fitness"] = self.compute_fitness(individual)
                individual["flag"] = 1

        self.population = sorted(self.population, key=lambda x: x["fitness"])
        fitness_list = [x["fitness"] for x in self.population]
        sum_fitness = sum(fitness_list)
        fit_pro = [fitness / sum_fitness for fitness in fitness_list]

        """ 按照概率分布选择出种群规模条染色体 """
        selected_population = np.random.choice(self.population, self.population_size, replace=False, p=fit_pro)
        print("  Min Fitness: %.2f  |  Max Fitness: %.2f  |  Average Fitness: %.2f" % (min(fitness_list), max(fitness_list), sum_fitness / len(fitness_list)))
        self.population = list(selected_population)

    def select_by_fitness(self) -> None:
        """
        优胜劣汰算法，直接选择适应值较大的染色体保留，该方法是简化版的选择方法，建议使用select()函数。
        @return: None
        """
        for count, individual in enumerate(self.population):
            count += 1
            if individual["flag"] == 0:
                individual["fitness"] = self.compute_fitness(individual)
                individual["flag"] = 1
        self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        self.population = self.population[:self.population_size]

    def get_offspring(self, parent: list) -> list:
        """
        根据两个父代个体（Individual）进行交叉后，返回两条新的子代个体（Individual）。
        @param parent: 父代 individual list
        @return: 子代 individual list
        """
        offspring = copy.deepcopy(parent)

        """ 随机选择交换基因段的左右位点索引（规则库染色体） """
        cross_left_position_rulebase = random.randint(0, self.rulebase_chromosome_size - 1)
        cross_right_position_rulebase = random.randint(cross_left_position_rulebase, self.rulebase_chromosome_size - 1)

        """ 交换子代对应位置的基因片段 """
        offspring[0]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1], \
            offspring[1]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1] = \
            offspring[1]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1], \
            offspring[0]["rulebase_chromosome"][cross_left_position_rulebase:cross_right_position_rulebase + 1]

        """ 随机选择交换基因段的左右位点索引（隶属函数染色体） """
        cross_left_position_mf = random.randint(0, self.mf_chromosome_size - 1)
        cross_right_position_mf = random.randint(cross_left_position_mf, self.mf_chromosome_size - 1)

        offspring[0]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1], \
            offspring[1]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1] = \
            offspring[1]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1], \
            offspring[0]["mf_chromosome"][cross_left_position_mf:cross_right_position_mf + 1]

        """ 新的子代没有被Simulation过，flag置为0，代表需要通过simulation后来计算适应值fitness """
        offspring[0]["flag"] = 0
        offspring[1]["flag"] = 0
        return offspring

    def cross(self) -> None:
        """
        将一个种群（population）中的所有个体（Individual）按概率进行交叉互换，并将子代添加入当前种群中。
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
        基因变异函数，对种群中每一个个体（Individual），随机选择染色体中的某一段，对该段中的每一个基因执行一次基因突变。
        @return: None
        """
        for individual in self.population:
            pro = random.random()
            if pro < self.mutation_pro:
                """ 选取突变点，并将该点之后的全部基因段进行基因突变（规则库） """
                mutation_pos_rulebase = random.randint(0, self.rulebase_chromosome_size - 1)
                gene_num = len(self.genes)
                individual["rulebase_chromosome"][mutation_pos_rulebase:] = [random.randint(0, gene_num - 1) for _ in
                                                                             range(
                                                                                 self.rulebase_chromosome_size - mutation_pos_rulebase)]

                """ 选取突变点，并将该点之后的全部基因段进行基因突变（隶属函数） """
                mutation_pos_mf = random.randint(0, self.mf_chromosome_size - 1)
                individual["mf_chromosome"][mutation_pos_mf:] = [random.randint(-10, 10) for _ in
                                                                 range(self.mf_chromosome_size - mutation_pos_mf)]
                """ 变异后的个体flag需被重置 """
                individual["flag"] = 0

    def get_optimal_individual(self):
        optimal_individual = self.population[0]
        for i in range(1, len(self.population) - 1):
            if self.population[i]['fitness'] > optimal_individual["fitness"]:
                optimal_individual = self.population[i]
        return optimal_individual

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

    def save_optimal_individual_to_file(self, path_rulebase, path_mf, optimal_individual):
        """
        将得分值最高的个体（Individual）存入文件中。
        @param path_rulebase: 规则库文件存放目录
        @param path_mf: 隶属函数参数文件存放目录
        @param optimal_individual: 最优个体对象
        @return: None
        """
        self.RB.encode_by_chromosome(optimal_individual["rulebase_chromosome"])
        self.RB.save_rule_base_to_file(path_rulebase)
        with open(path_mf, "w") as f:
            mf_dict = {"mf_offset": optimal_individual["mf_chromosome"]}
            json.dump(mf_dict, f)

    def train(self, save_best_individual_path="TrainedFile", save_all_path="SavedAllPopulation", base_path="models") -> None:
        """
        遗传算法训练函数。
        @param base_path: 模型存放总路径
        @param save_all_path: 种群存放路径
        @param save_best_individual_path: 最优个体存放路径
        @return: None
        """
        """ 若目录不存在则新建目录文件夹 """
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        save_best_individual_path = os.path.join(base_path, save_best_individual_path)
        save_all_path = os.path.join(base_path, save_all_path)
        if not os.path.exists(save_best_individual_path):
            os.mkdir(save_best_individual_path)
        if not os.path.exists(save_all_path):
            os.mkdir(save_all_path)

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
                os.path.join(save_best_individual_path, "RuleLib{}({}).json".format(count, optimal_individual["fitness"])),
                os.path.join(save_best_individual_path, "MF{}({}).json".format(count, optimal_individual["fitness"])),
                optimal_individual)

    @abstractmethod
    def start_simulation(self, simulator: DecisionSystemSimulation) -> float:
        """
        仿真器，用于根据FIS决策器的行为决策更新，获取并返回fitness。
        @param simulator: 继承自DecisionSystemSimulation的仿真器对象
        @return: 适应值
        """
        pass


if __name__ == '__main__':
    print("Hello GFS!")

