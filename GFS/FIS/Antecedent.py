from GFS.FIS.Term import Term
from GFS.FIS.Term import TermAggregate


class Antecedent(object):
    def __init__(self, clause):
        """
        模糊规则前件
        :param clause:规则前件，使用TermAggregate描述
        """
        self.clause = clause

    def antecedent_terms(self):
        terms = []

        def _find_terms(obj):
            if isinstance(obj, Term):
                terms.append(obj)
            elif obj is None:
                pass
            else:
                assert isinstance(obj, TermAggregate)
                _find_terms(obj.term1)
                _find_terms(obj.term2)

        _find_terms(self.antecedent)
        return terms

    def compute_value(self, input):
        """
        根据所有模糊变量的输入计算规则前件的值
        :param input: 所有模糊变量的数值,数据结构为字典，key为模糊变量label,value为具体数值
        :return: 该规则前件的值
        """
        return self.clause.compute_value(input)
