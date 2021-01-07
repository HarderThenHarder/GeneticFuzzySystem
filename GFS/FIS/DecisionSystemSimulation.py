import numpy as np
from .Term import Term
import math


class DecisionSystemSimulation(object):
    def __init__(self, decision_system):
        """
        使用相应的FIS决策系统进行仿真
        :param decision_system: 决策系统
        """
        self.decision_system = decision_system

    def simulation_get_action(self, input_info):
        """
        根据输入与规则库进行模糊推理，遍历整个规则库，找到最符合当前输入条件的规则，求出输出模糊变量具有最高置信度的模糊程度(Term)，将其索引作为决策行为索引。
        :param input_info: 输入模糊变量的观测数值
        :return: 决策行为索引
        """
        rule_lib = self.decision_system.rule_lib  # 先获得FIS的规则库
        output_fuzzy_variable = self.decision_system.fuzzy_variable_list[-1]  # 输出模糊变量
        action_value = [0 for _ in range(len(output_fuzzy_variable.terms))]   # 一个模糊程度（Term）对应一个action

        """ 遍历整个规则库，根据输入来判断当前状态（观测）最符合哪一条规则前件，继而计算输出模糊变量的哪一个 Term 置信度最高，置信度最高的 Term 则为最置信度最高的行为 """
        for rule in rule_lib:
            antecedent = rule.antecedent
            consequent = rule.consequent
            strength = antecedent.compute_value(input_info)  # 计算规则强度
            action_id = consequent.clause.id            # Term 的 id 即行为的 id
            if action_id == -1:
                continue
            """ 多个规则可能对应同一个 action(Term)，取置信度（规则强度）最大的 action（Term）Value 作为该行为的 action value """
            action_value[action_id] = max(action_value[action_id], strength)

        return action_value.index(max(action_value))    # 返回最高置信度行为的行为索引

    def simulation_get_crisp_value(self, input_info):
        """
        根据输入与规则库进行模糊推理，遍历整个规则库，找到最符合当前输入条件的规则，求出输出模糊变量具有最高置信度的模糊程度(Term)，并将该 Term 去模糊化成具体数值输出。
        @param input_info: 输入模糊变量的观测数值
        @return: consequent 经过去模糊化后的精确输出
        """
        rule_base = self.decision_system.rule_lib  # 先获得FIS的规则库
        output_fuzzy_variable = self.decision_system.fuzzy_variable_list[-1]  # 输出模糊变量
        action_num = len(output_fuzzy_variable.terms)
        output_terms_list = [Term for _ in range(action_num)]                 # 一个模糊程度（Term）对应一个action

        """ 将输出模糊变量的所有 Term 保存起来，一个 Term 对应一个 action """
        for k, v in output_fuzzy_variable.terms.items():
            output_terms_list[v.id] = v

        clip_value = [0 for _ in range(action_num)]

        """ 遍历整个规则库，保存输出模糊变量所有模糊程度（Term）的置信度（规则强度） """
        for rule in rule_base:
            antecedent = rule.antecedent
            consequent = rule.consequent
            strength = antecedent.compute_value(input_info)  # 计算规则强度
            action_id = consequent.clause.id
            if action_id == -1:
                continue
            clip_value[action_id] = max(clip_value[action_id], strength)

        """ 使用每一个 Term 的 clip_value（规则强度）进行插值计算，求出所有 Terms 的联合图形 """

        """ 用 clip 去截取三角隶属函数，交点的横坐标有可能是小数，但在 for 循环遍历横坐标时无法计算到小数，因此需手动求解交点横坐标 """
        interp_universe = []  # 用隶属值（clip value）去截三角形，得到两个交点的横坐标
        for term in output_fuzzy_variable.all_term():
            index = term.id
            clip = clip_value[index]
            a = term.trimf[0]
            b = term.trimf[1]
            c = term.trimf[2]
            if clip == 1:
                continue
            if a != b:
                interp_universe.append(clip * (b - a) + a)  # 利用相似三角形定理，求出左交点的横坐标：(x-a)/(b-a) = clip_v / 1
            if b != c:
                interp_universe.append(c - clip * (c - b))  # 利用相似三角形定理，求出右交点的横坐标：(c-x)/(c-b) = clip_v / 1

        """ 输出模糊变量原本各 Term 的 x 轴取值范围 """
        normal_universe = [i for i in range(output_fuzzy_variable.universe_down, output_fuzzy_variable.universe_up + 1)]
        """ 将交点横坐标并入到 Term 的 x 轴整数集中，为了方便后面 Mean of Max 方法计算最大 clip 值的横坐标集合 """
        final_universe = np.union1d(interp_universe, normal_universe)   # np.union1d 求并集

        """ 计算每一个 Term 在每一个横坐标点（整数点+两个交点）的隶属度  """
        mf_value = [[] for _ in range(action_num)]
        for index_universe in range(len(final_universe)):
            for index_action in range(action_num):
                mf_value[index_action].append(
                    output_terms_list[index_action].compute_membership_value(final_universe[index_universe]))

        """ 所有 Term 的联合最大隶属度图形（最大y值） """
        output_distribution = []
        for index_universe in range(len(final_universe)):
            Max = 0
            for index_action in range(action_num):
                mf_value[index_action][index_universe] = min(mf_value[index_action][index_universe],
                                                             clip_value[index_action])
                Max = max(Max, mf_value[index_action][index_universe])
            output_distribution.append(Max)

        """ 使用 mean of maximum 去模糊化分布，求所有对应y值等于最大y值的横坐标点的中心值 """
        Max = max(output_distribution)
        count = 0
        sum_count = 0
        for i in range(len(output_distribution)):
            if output_distribution[i] == Max:
                sum_count += final_universe[i]
                count += 1

        return sum_count / count
