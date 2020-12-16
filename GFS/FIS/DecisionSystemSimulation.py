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

    def simulation_get_action(self, input):
        """
        根据输入与规则库进行模糊推理，计算出相应的决策输出
        :param input: 输入模糊变量的具体数值
        :return: 输出动作的枚举ID
        """
        rule_base = self.decision_system.rule_lib # 先获得FIS的规则库
        fuzzy_variable_len = len(self.decision_system.fuzzy_variable_list)
        input_fuzzy_variable = self.decision_system.fuzzy_variable_list[0:fuzzy_variable_len - 1] #输入模糊变量
        output_fuzzy_variable = self.decision_system.fuzzy_variable_list[fuzzy_variable_len - 1] #输出模糊变量
        action_num = len(output_fuzzy_variable.terms)
        action_value = [0 for index in range(0, action_num)]
        for rule in rule_base:
            antecedent = rule.antecedent
            consequent = rule.consequent
            strength = antecedent.compute_value(input) #计算规则强度
            action_id = consequent.clause.id
            if action_id == -1:
                continue
            action_value[action_id] = max(action_value[action_id], strength)
        max_value = -1
        opt_action = -1
        for index in range(0, action_num):
            if action_value[index] > max_value:
                max_value = action_value[index]
                opt_action = index
        return opt_action

    def simulation_get_crisp_value(self, input):
        """
        根据输入与规则库计算精确数值输出
        @param input: 输入模糊变量的具体数值
        @return: consequent经过去模糊化后的精确输出
        """
        rule_base = self.decision_system.rule_lib  # 先获得FIS的规则库
        fuzzy_variable_len = len(self.decision_system.fuzzy_variable_list)
        input_fuzzy_variable_list = self.decision_system.fuzzy_variable_list[0:fuzzy_variable_len - 1]  # 输入模糊变量
        output_fuzzy_variable = self.decision_system.fuzzy_variable_list[fuzzy_variable_len - 1]  # 输出模糊变量
        action_num = len(output_fuzzy_variable.terms)
        term_temp = Term("temp", "temp", [0, 0, 0], -1)
        output_terms_list = [Term for index in range(0, action_num)]
        for k, v in output_fuzzy_variable.terms.items():
            output_terms_list[v.id] = v

        clip_value = [0 for index in range(0, action_num)]

        for rule in rule_base:
            antecedent = rule.antecedent
            consequent = rule.consequent
            strength = antecedent.compute_value(input)  # 计算规则强度
            action_id = consequent.clause.id
            if action_id == -1:
                continue
            clip_value[action_id] = max(clip_value[action_id], strength)
        mf_value = []
        # 使用每一个Term的clip_value进行插值，求出插值之后的universe
        interp_universe = []
        normal_universe = []
        # print(clip_value)
        for term in output_fuzzy_variable.all_term():
            id = term.id
            clip = clip_value[id]
            a = term.trimf[0]
            b = term.trimf[1]
            c = term.trimf[2]
            if math.fabs(clip - 1) < 0.000001:
                continue
            if a != b:
                interp_universe.append(clip * (b - a) + a)
            if b != c:
                interp_universe.append(c - clip * (c - b))

        for index in range(output_fuzzy_variable.universe_down, output_fuzzy_variable.universe_up + 1):
            normal_universe.append(index)
        final_universe = np.union1d(interp_universe, normal_universe)

        for index in range(0, action_num):
            mf_value.append([])
        for index_universe in range(0, len(final_universe)):
            for index_action in range(0, action_num):
                mf_value[index_action].append(output_terms_list[index_action].compute_membership_value(final_universe[index_universe]))

        output_distribution = []
        max_clip = max(clip_value)
        sum1 = 0
        count1 = 0
        # for index in range(0, action_num):
        #     if math.fabs(max_clip - clip_value[index]) < 0.000001:
        #         for i in range(0, len(mf_value[index])):
        #             if i - 1 >= 0:
        #                 d1 = mf_value[index][i] - max_clip
        #                 d2 = mf_value[index][i - 1] - max_clip
        #                 if d1 >= 0.000001 and d2 <= 0.000001:
        #                     inter_value = i - (mf_value[index][i] - max_clip) / (mf_value[index][i] - mf_value[index][i - 1])
        #                     sum1 += inter_value
        #                     count1 += 1
        #                 elif d1 <= 0.00001 and d2 >= 0.000001:
        #                     inter_value = (i - 1) + (mf_value[index][i - 1] - max_clip) / (mf_value[index][i - 1] - mf_value[index][i])
        #                     sum1 += inter_value
        #                     count1 += 1

        for index_universe in range(0, len(final_universe)):
            Max = 0
            for index_action in range(0, action_num):
               mf_value[index_action][index_universe] = min(mf_value[index_action][index_universe], clip_value[index_action])
               Max = max(Max, mf_value[index_action][index_universe])
            output_distribution.append(Max)

        #使用mean of maximum去模糊化分布
        Max = 0
        for value in output_distribution:
            Max = max(Max, value)
        count = 0
        sum = 0
        for i in range(0, len(output_distribution)):
            if math.fabs(Max - output_distribution[i]) <= 0.000001:
                sum += final_universe[i]
                count += 1
        sum += sum1
        count += count1

        return sum / count
