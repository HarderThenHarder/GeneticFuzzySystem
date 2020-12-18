import json
import random
import copy
from GFS.FIS.Antecedent import Antecedent
from GFS.FIS.Consequent import Consequent
from GFS.FIS.Rule import Rule


class RuleLib(object):
    def __init__(self, fuzzy_variable_list):
        self.rule_lib = None
        self.chromosome = None
        self.fuzzy_variable_list = copy.deepcopy(fuzzy_variable_list)
        self.generate_random_rule_base(len(self.fuzzy_variable_list) - 1)   # 初始化规则库，默认规则前件个数等于模糊变量总长度-1
        self.count = 0

    def encode(self, rules):
        """
        将Rule集进行编码，以便表示与保存
        :param rules: Rule集合
        :return:
        """
        rule_lib = []
        chromosome = []
        for rule in rules:
            antecedent_terms = rule.antecedent.clause.antecedent_terms()
            consequent_term = rule.consequent.clause
            rule_code = []
            for term in antecedent_terms:
                rule_code.append(term.id)
            rule_code.append(consequent_term.id)
            chromosome.append(consequent_term.id)
            rule_lib.append(rule_code)
        self.rule_lib = rule_lib
        self.chromosome = chromosome

    def encode_by_chromosome(self, chromosome) -> None:
        """
        通过染色体基因还原整个规则库，由 [a1, a2, a3, ...] 变成 [[condition1, a1], [condition2, a2], ...]。
        @param chromosome: 染色体组
        @return: None
        """
        rule_code = []
        chromosome_code = []

        length = len(self.fuzzy_variable_list) - 1
        self.count = 0

        def dfs(depth, pre):
            up = len(self.fuzzy_variable_list[depth].all_term())
            for i in range(up):
                now_pre = copy.deepcopy(pre)
                now_pre.append(i)
                if depth != length - 1:
                    dfs(depth + 1, now_pre)
                else:
                    consequent = chromosome[self.count]
                    self.count += 1
                    now_pre.append(consequent)
                    rule_code.append(now_pre)
                    chromosome_code.append(consequent)

        dfs(0, [])
        self.rule_lib = copy.deepcopy(rule_code)
        self.chromosome = copy.deepcopy(chromosome_code)

    def decode(self) -> list:
        """
        将编码表示的规则库解析成FIS系统中的Rule对象。
        @return: 包含N个Rule对象的规则列表
        """
        rules = []
        rule_len = len(self.fuzzy_variable_list)
        for rule in self.rule_lib:
            term_list = []
            for index in range(rule_len):
                all_terms = self.fuzzy_variable_list[index].all_term()
                if rule[index] == -1:
                    term_index = len(all_terms) - 1
                else:
                    term_index = rule[index]
                term_list.append(all_terms[term_index])
            clause = term_list[0]
            for i in range(1, (rule_len - 1)):
                clause = clause & term_list[i]
            antecedent_clause = clause
            consequent_clause = term_list[rule_len - 1]
            antecedent = Antecedent(antecedent_clause)
            consequent = Consequent(consequent_clause)
            now_rule = Rule(antecedent, consequent)
            rules.append(now_rule)
        return rules

    def generate_random_rule_base(self, length):
        """
        通过随机生成规则来填满规则库。
        @param length: 有多少个规则前件
        @return: None
        """
        rule_code = []
        chromosome_code = []

        def dfs(depth, pre):
            up = len(self.fuzzy_variable_list[depth].all_term())
            for i in range(up):
                now_pre = copy.deepcopy(pre)
                now_pre.append(i)
                if depth != length - 1:
                    dfs(depth + 1, now_pre)
                else:
                    consequent = random.randint(0, len(self.fuzzy_variable_list[depth + 1].all_term())) - 1
                    now_pre.append(consequent)
                    rule_code.append(now_pre)
                    chromosome_code.append(consequent)
        dfs(0, [])
        self.rule_lib = copy.deepcopy(rule_code)
        self.chromosome = copy.deepcopy(chromosome_code)

    def load_rule_base_from_file(self, filepath):
        """
        从文件中加载规则库
        filepath:规则文件存放路径
        :return:
        """
        with open(filepath, "r") as f:
            rule_base_dict = json.load(f)
            self.rule_lib = rule_base_dict["RuleLib"]
            self.chromosome = rule_base_dict["chromosome"]

    def save_rule_base_to_file(self, filepath):
        """
        将规则库保存成本地文件
        :param filepath: 保存文件的路径
        :return: None
        """
        with open(filepath, "w") as f:
            rule_base_dict = {'RuleLib': self.rule_lib, 'chromosome': self.chromosome}
            json.dump(rule_base_dict, f)

    def save_mf_to_file(self, filepath, optimal_individual):
        """
        将隶属函数参数保存成本地文件
        @param optimal_individual: 最优个体对象
        @param filepath: 保存文件路径
        @return: None
        """
        with open(filepath, "w") as f:
            mf_dict = {"mf_offset": optimal_individual["mf_chromosome"]}
            json.dump(mf_dict, f)
