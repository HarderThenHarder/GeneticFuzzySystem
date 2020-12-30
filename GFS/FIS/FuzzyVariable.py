from collections import OrderedDict
from .Term import Term
import numpy as np
import matplotlib.pyplot as plt


class FuzzyVariable(object):
    def __init__(self, universe, label, defuzzify_method='mom'):
        """
        模糊变量：用于FIS进行决策的输入、输出变量
        :param label: 变量的名称
        :param universe: 变量的取值范围
        :param terms: 该变量可以模糊化为哪些类别
        """
        self.label = label
        self.universe_down = universe[0]
        self.universe_up = universe[1]
        self.terms = OrderedDict()

    def all_term(self):
        all_term = []
        for k, v in self.terms.items():
            all_term.append(v)
        return all_term

    def __getitem__(self, key):
        """
        可以使用variable["label"]访问“label”term
        :param key: term的label
        :return: 对应的term
        """
        if key in self.terms.keys():
            return self.terms[key]
        else:
            # Build a pretty list of available mf labels and raise an
            # informative error message
            options = ''
            i0 = len(self.terms) - 1
            i1 = len(self.terms) - 2
            for i, available_key in enumerate(self.terms.keys()):
                if i == i1:
                    options += "'" + str(available_key) + "', or "
                elif i == i0:
                    options += "'" + str(available_key) + "'."
                else:
                    options += "'" + str(available_key) + "'; "
            raise ValueError("Membership function '{0}' does not exist for "
                             "{1} {2}.\n"
                             "Available options: {3}".format(
                                 key, self.__name__, self.label, options))

    def __setitem__(self, key, item):
        """
        可以使用variable["new_label"] = new_term为模糊变量设置新的item
        :param key:term的label
        :param item:新的term实例
        :return:
        """
        if isinstance(item, Term):
            if item.label != key:
                raise ValueError("Term's label must match new key")
            if item.parent is None:
                raise ValueError("Term must not already have a parent")
        else:
            raise ValueError("Unknown Type")
        self.terms[key] = item

    def automf(self, number=5, variable_type='quantity', discrete=False, names=None, special_case=False, special_mf_abc=None, special_case_name="special_case"):
        """
        生成指定数量的Term，其隶属函数默认为三角型隶属函数
        @param discrete: 该变量是否为离散型变量，如代表类别型的变量只能取（1，2，3），不能取 1.33
        @param number: 生成的Term数量（不包括特殊隶属类），默认值为5
        @param variable_type: 质量型或者数量型(quality, quantity)
        @param names: 相应Term的名字，其数量应该与number相同
        @param special_mf_abc: 特殊隶属类的三角形顶点值
        @param special_case: 是否自动分配特殊隶属类
        @param special_case_name: 特殊隶属类的名称，默认为 special_case
        """
        if not discrete:
            if names is not None:
                # set number based on names passed
                number = len(names)
            else:
                if number not in [3, 5, 7]:
                    raise ValueError("If number is not 3, 5, or 7, "
                                     "you must pass a list of names "
                                     "equal in length to number.")

                if variable_type.lower() == 'quality':
                    names = ['dismal',
                             'poor',
                             'mediocre',
                             'average',
                             'decent',
                             'good',
                             'excellent']
                else:
                    names = ['lowest',
                             'lower',
                             'low',
                             'average',
                             'high',
                             'higher',
                             'highest']

                if number == 3:
                    if variable_type.lower() == 'quality':
                        names = names[1:6:2]
                    else:
                        names = names[2:5]
                if number == 5:
                    names = names[1:6]

            limits = [self.universe_down, self.universe_up]
            universe_range = limits[1] - limits[0]
            widths = [universe_range / ((number - 1) / 2.)] * int(number)
            centers = np.linspace(limits[0], limits[1], number)

            abcs = [[c - w / 2, c, c + w / 2] for c, w in zip(centers, widths)]

            # Clear existing adjectives, if any
            self.terms = OrderedDict()

            """ 若存在特殊隶属类，则将特殊隶属类添加到最后一项 """
            if special_case:
                assert special_mf_abc, "[ERROR] You must specify the special_mf_abc value for special mf function."
                names.append(special_case_name)
                abcs.append(special_mf_abc)

            # Repopulate
            index = 0
            for name, abc in zip(names, abcs):
                term = Term(name, self.label, abc, index)
                index += 1
                self[name] = term
        else:
            abcs = [[0, 0, 0] for _ in range(number)]

            index = 0
            for abc in abcs:
                name = str(index)
                term = Term(name, self.label, abc, index)
                index += 1
                self[name] = term

            if special_case:
                self[special_case_name] = Term(special_case_name, self.label, [0, 0, 0], index)

    def show(self):
        color = ['red', 'green', 'blue']
        for k, v in self.terms.items():
            plt.plot(v.trimf, [0, 1, 0], color=color[v.id], label=v.label)
        plt.legend(loc="best")
        plt.xlabel(self.label)
        plt.ylabel("membership")
        plt.show()
