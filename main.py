"""
@Author: P_k_y
@Time: 2020/12/16
"""
from GFS.FIS.DecisionSystemSimulation import DecisionSystemSimulation
from GFS.FIS.FuzzyVariable import FuzzyVariable
from GFS.FIS.RuleLib import RuleLib
from GFS.FIS.Term import Term
from GFS.GeneticFuzzySystem import BaseGFS
import random


class GFS(BaseGFS):

    def __init__(self, rule_lib, population_size, episode, mutation_pro=0.01, cross_pro=0.9):
        """
        实现自定义GFS子类（继承自BaseGFS基类）并实现自定义计算仿真方法。
        @param rule_lib: 规则库对象
        @param population_size: 种群规模（存在的染色体条数，可以理解为存在的规则库个数）
        @param episode: 训练多少轮
        @param mutation_pro: 变异概率
        @param cross_pro: 交叉概率
        """
        super().__init__(rule_lib, population_size, episode, mutation_pro, cross_pro)

    """ 实现父类抽象方法 """
    def start_simulation(self, simulators: list) -> float:
        """
        根据指定的simulator列表计算出一次仿真后的fitness。
        @param simulators: DecisionSystemSimulation对象列表
        @return: 返回fitness值
        """
        return random.randint(0, 1000)


if __name__ == '__main__':
    quality = FuzzyVariable([0, 10], 'quality')
    servive = FuzzyVariable([0, 10], 'service')
    tip = FuzzyVariable([0, 25], 'tip')

    quality.automf(3)
    servive.automf(3)

    tip['low'] = Term('low', 'tip', [-13, 0, 13], 0)
    tip['medium'] = Term('medium', 'tip', [0, 13, 25], 1)
    tip['high'] = Term('high', 'tip', [13, 25, 38], 2)
    test_rb = RuleLib([quality, servive, tip])
    ga_test = GFS(rule_lib=test_rb, population_size=6, episode=50, mutation_pro=0.01, cross_pro=0.9)
    ga_test.train()

