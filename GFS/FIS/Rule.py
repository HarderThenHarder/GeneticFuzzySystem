class Rule(object):
    def __init__(self, antecedent, consequent):
        """
        模糊规则定义，每一条模糊规则由规则前件与规则结果组成
        :param antecedent: 规则前件
        :param consequent: 规则结果
        """
        self.antecedent = antecedent
        self.consequent = consequent

    def get_rule_str(self):
        antecedent_string = self.antecedent.clause.get_string()
        consequent_string = self.consequent.clause.parent + ' is ' + self.consequent.clause.label
        rule_string = antecedent_string + ' THEN ' + consequent_string
        return rule_string
