class DecisionSystem(object):
    def __init__(self, fuzzy_variable_list, rule_lib):
        """
        FIS决策系统：由模糊变量集与模糊规则库组成
        :param fuzzy_variable_list: 模糊变量集
        :param rule_lib: 模糊规则库
        """
        self.fuzzy_variable_list = fuzzy_variable_list
        self.rule_lib = rule_lib
