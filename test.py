import skfuzzy as fuzzy
import numpy as np
import skfuzzy.control as ctrl


def transfer_number_to_class_index(value, max_value, class_num) -> int:
    """
    给定一个输入值，判断该值处于一个连续区间中的第几段，用于将连续变量离散化。
    :param value: 输入值
    :param max_value: 区间最大值
    :param class_num: 该区间一共被分成几段
    :return: 所在段索引
    """
    assert value <= max_value, "[ERROR] Input Value(%f) can't be bigger than Max Value(%f)!" % (value, max_value)
    block_length = max_value / class_num
    if value == 0:
        return 0
    else:
        return int((value - 1e-5) / block_length)


def main():
    """ 建立模糊变量 """
    radar_heading = ctrl.Antecedent(np.arange(0, 360, 1), 'radar_heading')
    radar_distance = ctrl.Antecedent(np.arange(0, 100, 1), 'radar_distance')
    action = ctrl.Consequent(np.arange(0, 3, 1), 'action')

    """ 建立隶属函数 """
    radar_heading.automf(3, names=[str(x) for x in range(3)])
    radar_distance.automf(3, names=['near', 'middle', 'far'])
    action.automf(3, names=['left', 'front', 'right'])

    """ 建立规则库 """
    rule_list = [ctrl.Rule(radar_heading['0'] | radar_distance['near'], action['left']),
                 ctrl.Rule(radar_heading['1'] | radar_distance['middle'], action['front']),
                 ctrl.Rule(radar_heading['2'] | radar_distance['far'], action['right'])]
    rule_list[0].view()

    """ 建立FIS仿真器（用于进行计算） """
    action_simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rule_list))

    """ 输入并进行判断输出结果 """
    action_simulation.inputs({
        'radar_heading': 103,
        'radar_distance': 23
    })
    action_simulation.compute()

    print(action_simulation.output)
    action_value = action_simulation.output['action']
    action_idx = transfer_number_to_class_index(action_value, action.universe.max(), action.universe.shape[0])
    print("Action Index After Transformed: ", action_idx)
    action.view(sim=action_simulation)


if __name__ == '__main__':
    main()
