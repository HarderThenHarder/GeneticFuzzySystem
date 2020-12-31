"""
@Author: P_k_y
@Time: 2020/12/16
"""
from GFS.FIS.FuzzyVariable import FuzzyVariable
from GFS.FIS.RuleLib import RuleLib
from GFS.GeneticFuzzySystem import BaseGFT
import gym


class GFT(BaseGFT):

    def __init__(self, rule_lib_list, population_size, episode, mutation_pro, cross_pro, simulator, parallelized):
        """
        实现自定义GFT子类（继承自BaseGFT基类）并实现自定义计算仿真方法。
        @param rule_lib_list: 规则库对象
        @param population_size: 种群规模（存在的染色体条数，可以理解为存在的规则库个数）
        @param episode: 训练多少轮
        @param mutation_pro: 变异概率
        @param cross_pro: 交叉概率
        @param simulator: 仿真器对象，用于获取观测和回报
        @param parallelized: 是否启用多进程并行计算
        """
        super().__init__(rule_lib_list, population_size, episode, mutation_pro, cross_pro, simulator, parallelized)

    """ 实现父类抽象方法 """
    def start_simulation(self, controllers: list, simulator) -> float:
        """
        自定义 GFT 算法模块与仿真器 Simulator（gym） 之间的数据交互过程，返回仿真器的 reward 值。
        @param simulator: 仿真器对象
        @param controllers: 控制器列表，一个controller决策一个行为。
        @return: fitness
        """
        controller = controllers[0]
        fitness = 0

        obs_list = simulator.reset()
        for _ in range(1000):

            # simulator.render()

            """ CartPole-v0 中共包含 4 个观测，在FIS决策器中需要对应拆分成 4 个模糊变量输入 """
            obs_input = {
                "obs1": obs_list[0],
                "obs2": obs_list[1],
                "obs3": obs_list[2],
                "obs4": obs_list[3]
            }

            action = controller.simulation_get_action(obs_input)    # 利用 FIS 决策器获得行为决策
            obs_list, r, done, _ = simulator.step(action)
            fitness += r

            """ Reward Shaping: 若杆子与垂直面夹角越小则得分越高 """
            angle = abs(obs_list[2])
            r_shaping = (0.418 - angle) / 0.418

            fitness += r_shaping

            if done:
                break

        return fitness


def create_gft(simulator) -> GFT:
    """
    建立GFT对象，根据具体场景建立模糊变量与规则库。
    @return: GFT对象
    """

    """ 1. 构建模糊变量，采用 gym 中 CartPole-v0 作为示例，共包含 4 个观测输入，1 个行为输出 """
    obs1 = FuzzyVariable([-4.9, 4.9], "obs1")
    obs2 = FuzzyVariable([-3.40e+38, 3.40e+38], "obs2")
    obs3 = FuzzyVariable([-0.418, 0.418], "obs3")
    obs4 = FuzzyVariable([-4.18e-01, 4.18e-01], "obs4")

    action = FuzzyVariable([0, 1], "action")

    """ 2. 为模糊变量分配隶属函数 """
    obs1.automf(5)
    obs2.automf(5)
    obs3.automf(5)
    obs4.automf(5)
    action.automf(2, discrete=True)     # 行为输出是离散型的模糊变量

    """ 3. 构建 RuleLib 规则库 """
    controller = RuleLib([obs1, obs2, obs3, obs4, action])

    """ 4. 构建 GFT 对象 """
    return GFT(rule_lib_list=[controller], population_size=40, episode=200, mutation_pro=0.1, cross_pro=0.9,
               simulator=simulator, parallelized=False)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    gft = create_gft(env)
    gft.train()
    # gft.evaluate("models/OptimalIndividuals/[Epoch_21]Individual(323.3).json")
