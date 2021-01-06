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

        """ 使用每一个 Term 的 clip_value（规则强度）进行插值，求出插值之后的 universe """
        mf_value = []
        interp_universe = []
        normal_universe = []

        """ 根据之前计算出的每一个 Term 在 x 轴取值范围的并集"""
        for term in output_fuzzy_variable.all_term():
            id = term.id
            clip = clip_value[id]
            a = term.trimf[0]
            b = term.trimf[1]
            c = term.trimf[2]
            if math.fabs(clip - 1) < 1e-6:
                continue
            if a != b:
                interp_universe.append(clip * (b - a) + a)
            if b != c:
                interp_universe.append(c - clip * (c - b))

        """ 输出模糊变量原本各 Term 的 x 轴取值范围 """
        for index in range(output_fuzzy_variable.universe_down, output_fuzzy_variable.universe_up + 1):
            normal_universe.append(index)
        final_universe = np.union1d(interp_universe, normal_universe)   # np.union1d 求并集

        for index in range(action_num):
            mf_value.append([])

        """ 根据每一个 Term（Action）的隶属函数值  """
        for index_universe in range(len(final_universe)):
            for index_action in range(action_num):
                mf_value[index_action].append(
                    output_terms_list[index_action].compute_membership_value(final_universe[index_universe]))

        output_distribution = []

        for index_universe in range(len(final_universe)):
            Max = 0
            for index_action in range(action_num):
                mf_value[index_action][index_universe] = min(mf_value[index_action][index_universe],
                                                             clip_value[index_action])
                Max = max(Max, mf_value[index_action][index_universe])
            output_distribution.append(Max)

        """ 使用 mean of maximum 去模糊化分布 """
        Max = max(output_distribution)
        count = 0
        sum_count = 0
        for i in range(len(output_distribution)):
            if math.fabs(Max - output_distribution[i]) <= 1e-6:
                sum_count += final_universe[i]
                count += 1

        return sum_count / count
