import numpy as np
import pandas as pd

data = pd.read_csv('PEMS07.csv')
s1 = data['from'].tolist()
s2 = data['to'].tolist()
cost = data['cost'].tolist()
num = len(s1)
weight = np.full(shape=[883, 883], fill_value=100, dtype=float)
for i in range(num):
    weight[s1[i]][s2[i]] = cost[i]
    weight[s2[i]][s1[i]] = cost[i]

for i in range(883):
    weight[i][i] = 0.0

#dis_csv = pd.DataFrame(weight)
#dis_csv.to_csv('07_mat.csv', header=None, index=None)

# 弗洛伊德算法

class Graph(object):
    def __init__(self, length: int, matrix: [], vertex: []):
        """
        :param length: 大小
        :param matrix: 邻接矩阵
        :param vertex: 顶点数组
        """
        # 保存，从各个顶点出发到其它顶点的距离，最后的结果，也保留在该数组
        self.dis = matrix
        # 保存到达目标顶点的前驱顶点
        self.pre = [[0 for col in range(length)] for row in range(length)]
        self.vertex = vertex
        # 对 pre数组进行初始化，存放的是前驱顶点的下标
        for i in range(length):
            for j in range(length):
                self.pre[i][j] = i

    # 显示pre数组和dis数组
    def save_graph(self):
        dis_csv = pd.DataFrame(self.dis)
        dis_csv.to_csv('07_weight.csv', header=None, index=None)

    # 佛洛依德算法
    def floyd(self):
        length: int = 0  # 变量保存距离
        # 对中间顶点的遍历,k 就是中间顶点的下标
        for k in range(len(self.dis)):  # ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            # 从 i顶点开始出发['A', 'B', 'C', 'D', 'E', 'F', 'G']
            for i in range(len(self.dis)):
                # 到达j顶点 ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                for j in range(len(self.dis)):
                    length = self.dis[i][k] + self.dis[k][j]  # 求出从i 顶点出发，经过k中间顶点，到达j顶点距离
                    if length < self.dis[i][j]:  # 如果length 小于dis[i][j]
                        self.dis[i][j] = length  # 更新距离
                        self.dis[j][i] = length
                        self.pre[i][j] = self.pre[k][j]  # 更新前驱顶点
                        self.pre[j][i] = self.pre[i][k]


# 顶点数组
vertex = list(range(883))
g = Graph(len(vertex), weight, vertex)
# 调用弗洛伊德算法
g.floyd()
g.save_graph()
