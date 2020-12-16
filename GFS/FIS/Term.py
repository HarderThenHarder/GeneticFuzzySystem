import numpy as np


class TermPrimitive(object):
    def membership_value(self):
        raise NotImplementedError("Implement in concrete class")

    def __and__(self, other):
        if not isinstance(other, TermPrimitive):
            raise ValueError("Can only construct 'AND' from the term of a fuzzy variable")
        return TermAggregate(self, other, 'and')

    def __or__(self, other):
        if not isinstance(other, TermPrimitive):
            raise ValueError("Can only construct 'OR' form the term of a fuzzy variable")
        return TermAggregate(self, other, 'or')

    def __invert__(self):
        return TermAggregate(self, None, 'not')


class Term(TermPrimitive):
    def __init__(self, label, parent, trimf, id):
        self.label = label #标签
        self.parent = parent #标签的父变量
        self.trimf = trimf #三角形隶属函数的三个顶点
        self.span = max(trimf[1] - trimf[0], trimf[2] - trimf[1])
        self.id = id #term编号

    def compute_membership_value(self, value):
        """
        根据隶属函数计算变量隶属于这一term的隶属度
        :param value: 变量的值
        :return: 变量隶属该Term的隶属度
        """
        a = self.trimf[0]
        b = self.trimf[1]
        c = self.trimf[2]
        if a == b and b == c and a == 0:# 类别型模糊变量
            if value == self.id:
                return 1
            else:
                return 0
        if a == b:
            if value >= c:
                return 0
            elif value < a:
                return 0
            else:
                return (c - value) / (c - b)
        if b == c:
            if value <= a:
                return 0
            elif value >= c:
                return 1
            else:
                return (value - a) / (b - a)
        # 普通三角型隶属函数
        if value < a:
            return 0
        elif value > c:
            return 0
        elif value < b:
            return (value - a) / (b - a)
        else:
            return (c - value) / (c - b)

    def compute_value(self, value):
        return self.compute_membership_value(value[self.parent])


class FuzzyAggregationMethods(object):
    def __init__(self, and_func=np.fmin, or_func=np.fmax):
        # Default and to OR = max and AND = min
        self.and_func = and_func
        self.or_func = or_func


class TermAggregate(TermPrimitive):
    """
    使用AND，OR连接两个TermPrimitive
    """

    def __init__(self, term1, term2, kind):
        assert isinstance(term1, TermPrimitive)
        if kind in ('and', 'or'):
            assert isinstance(term2, TermPrimitive)
        elif kind == 'not':
            assert term2 is None, "NOT (~) operates on a single Term, not two."
        else:
            raise ValueError("Unexpected kind")

        self.term1 = term1
        self.term2 = term2
        self.kind = kind
        self._agg_methods = FuzzyAggregationMethods()

    @property
    def agg_methods(self):
        return self._agg_methods

    @agg_methods.setter
    def agg_methods(self, agg_methods):
        if not isinstance(agg_methods, FuzzyAggregationMethods):
            raise ValueError("Expected FuzzyAggregationMethods")
        self._agg_methods = agg_methods

        # 将聚合方法向下传递给TermPrimitive
        for term in (self.term1, self.term2):
            if isinstance(term, TermAggregate):
                term.agg_methods = agg_methods

    def compute_value(self, input):
        """
        根据具体模糊变量的数值计算该TermAggregate的数值
        :param input:输入字典
        :return:
        """
        term1_value = None
        term2_value = None
        if isinstance(self.term1, Term):
            term1_value = self.term1.compute_membership_value(input[self.term1.parent])
        elif isinstance(self.term1, TermAggregate):
            term1_value = self.term1.compute_value(input)

        if isinstance(self.term2, Term):
            term2_value = self.term2.compute_membership_value(input[self.term2.parent])
        elif isinstance(self.term2, TermAggregate):
            term2_value = self.term2.compute_value(input)

        if self.kind == 'and':
            return self._agg_methods.and_func(term1_value, term2_value)
        if self.kind == 'or':
            return self._agg_methods.or_func(term1_value, term2_value)

    def get_string(self):

        term1_string = ''
        term2_string = ''
        if isinstance(self.term1, Term):
            term1_string = self.term1.parent + ' is ' + self.term1.label
        elif isinstance(self.term1, TermAggregate):
            term1_string = self.term1.get_string()

        if isinstance(self.term2, Term):
            term2_string = self.term2.parent + ' is ' + self.term2.label
        elif isinstance(self.term2, TermAggregate):
            term2_string = self.term2.get_string()

        if self.kind == 'and':
            return term1_string + ' AND ' + term2_string
        if self.kind == 'or':
            return term1_string + ' OR ' + term2_string

