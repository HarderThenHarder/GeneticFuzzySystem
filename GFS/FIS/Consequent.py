class Consequent(object):
    def __init__(self, clause):
        """
        模糊规则结果，使用TermAggregate描述
        :param clause: 规则结果(一般使用一个Term表示)
        """
        self.clause = clause