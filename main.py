"""
@Author: P_k_y
@Time: 2020/12/16
"""
from GFS.FIS.DecisionSystemSimulation import DecisionSystemSimulation
from GFS.FIS.FuzzyVariable import FuzzyVariable
from GFS.FIS.RuleLib import RuleLib
from GFS.FIS.Term import Term
from GFS.GeneticFuzzySystem import BaseGFS, BaseGFT
import random


class GFS(BaseGFS):

    def __init__(self, rule_lib, population_size, episode, mutation_pro=0.01, cross_pro=0.9, simulator=None):
        """
        实现自定义GFS子类（继承自BaseGFS基类）并实现自定义计算仿真方法。
        @param rule_lib: 规则库对象
        @param population_size: 种群规模（存在的染色体条数，可以理解为存在的规则库个数）
        @param episode: 训练多少轮
        @param mutation_pro: 变异概率
        @param cross_pro: 交叉概率
        """
        super().__init__(rule_lib, population_size, episode, mutation_pro, cross_pro, simulator)

    """ 实现父类抽象方法 """
    def start_simulation(self, simulator: DecisionSystemSimulation) -> float:
        """
        根据指定的simulator列表计算出一次仿真后的fitness。
        @param simulator: DecisionSystemSimulation对象
        @return: 返回fitness值
        """
        return random.randint(0, 1000)


def test_GFS():
    """
    GFS类测试函数。
    @return: None
    """
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


class GFT(BaseGFT):

    def __init__(self, rule_lib_list, population_size, episode, mutation_pro=0.01, cross_pro=0.9, simulator=None):
        """
        实现自定义GFT子类（继承自BaseGFT基类）并实现自定义计算仿真方法。
        @param rule_lib_list: 规则库对象
        @param population_size: 种群规模（存在的染色体条数，可以理解为存在的规则库个数）
        @param episode: 训练多少轮
        @param mutation_pro: 变异概率
        @param cross_pro: 交叉概率
        @param simulator: 仿真器对象，用于获取观测和回报
        """
        super().__init__(rule_lib_list, population_size, episode, mutation_pro, cross_pro, simulator)

    """ 实现父类抽象方法 """
    def start_simulation(self, simulators: list) -> float:
        """
        根据指定的simulator列表计算出一次仿真后的fitness。
        @param simulators: DecisionSystemSimulation对象列表
        @return: 返回fitness值
        """
        return random.randint(0, 1000)


def test_GFT():
    """
    GFT类测试函数。
    @return: None
    """
    quality = FuzzyVariable([0, 10], 'quality')
    service = FuzzyVariable([0, 10], 'service')
    tip = FuzzyVariable([0, 25], 'tip')

    quality.automf(3)
    service.automf(3)

    tip['low'] = Term('low', 'tip', [-13, 0, 13], 0)
    tip['medium'] = Term('medium', 'tip', [0, 13, 25], 1)
    tip['high'] = Term('high', 'tip', [13, 25, 38], 2)

    """ 传入一个规则库列表，代表存在多个规则库，一个规则库决策一种特定行为 """
    rule_lib_list = [RuleLib([quality, service, tip]), RuleLib([quality, service, tip])]
    ga_test = GFT(rule_lib_list=rule_lib_list, population_size=6, episode=10, mutation_pro=0.99, cross_pro=0.9)
    ga_test.train()


if __name__ == '__main__':
    # test_GFS()
    test_GFT()
