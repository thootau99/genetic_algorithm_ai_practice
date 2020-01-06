
# ravel -> 2d-array -> 1d-array
# exp -> Calculate the exponential of all elements in the input array.
import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 20  # DNA size
CROSS_RATE = 0.2
MUTATE_RATE = 0.05
POP_SIZE = 200
N_GENERATIONS = 500
RAIN_RATE = 0.1


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, rain_rate):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.pop = np.vstack([np.random.permutation(DNA_size)
                              for _ in range(pop_size)])
        self.rain_rate = rain_rate

    # get cities' coord in order
    def translateDNA(self, DNA, city_position, priority, terrain):
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        prio = np.empty_like(DNA, dtype=np.int)
        terr = np.empty_like(DNA, dtype=np.int)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            # city_coord 內中全部行的第一個元素填到 line_x[i, :]
            line_x[i, :] = city_coord[:, 0]
            # city_coord 內中全部行的第二個元素甜到 line_y[i. :]
            line_y[i, :] = city_coord[:, 1]
            # 結論：len(city_coord[:]) == 2,  分別儲存x, y
            prio[i, :] = self.priorityCalc(d, priority)
            terr[i] = self.terrainCalc(d, terrain)
        return line_x, line_y, prio, terr

    def terrainCalc(self, DNA, terrain):
        result = []
        currentState = False
        for i, d in enumerate(DNA):
            if i == 0:
                currentState = terrain[d]
                result.append(0)
            else:
                if currentState == True and terrain[d] == True:
                    result.append(0)  # 平的移動
                elif currentState == True and terrain[d] == False:
                    result.append(1)  # 下山
                elif currentState == False and terrain[d] == True:
                    result.append(2)  # 上山
                elif currentState == False and terrain[d] == False:
                    result.append(0)  # 平的移動
                currentState = terrain[d]
        return result

    def priorityCalc(self, DNA, priority):
        priority_copy = priority.copy()  # copy 一份優先權清單
        score = 0  # 初始化積分
        for i in DNA:
            if priority_copy[i] > 0:  # 假如積分不小於零就加
                score = score + priority_copy[i]
            priority_copy[:] = priority_copy[:] - 1  # 跑過一輪後全部的優先權 -1
        return score

    def get_fitness(self, line_x, line_y, priority, terrain):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        prio = np.empty((priority.shape[0],), dtype=np.float64)  # 產生陣列放優先權積分
        terr = np.empty((priority.shape[0],), dtype=np.float64)
        weather_calc = np.random.randint(0, 2, self.pop_size).astype(
            np.bool)  # 產生是否為雨天
        for i, (xs, ys, pr) in enumerate(zip(line_x, line_y, priority)):
            total_distance[i] = np.sum(
                np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
            # 計算距離
            # 用 diff 來xs[index+1] - xs[index]... 之類的方式來變魔術
            # 用 square 來做平方(整個陣列都平方)
            # 用 sqrt 來開根號 (畢氏定理啦)
            # 用 sum 把全部的點的距離連起來

            if not weather_calc[i]:
                prio[i] = total_distance[i] * \
                    ((1.0 - (0.05 * pr[0])))  # 優先權積分越高在距離上會有減免（積分*5%）
            else:
                prio[i] = total_distance[i] * \
                    ((1.0 - (0.05 * pr[0]))) * \
                    1.005  # 優先權積分越高在距離上會有減免（積分*5%）(雨天模式，距離*1.1)
        for i, d in enumerate(zip(terrain)):
            one = 0
            two = 0
            for j in d:
                for _j in j:
                    if _j == 1:
                        one = one + 1
                    elif _j == 2:
                        two = two + 1
            result = 1 + pow(one, 1.12) * pow(two, 0.18)
            prio[i] = prio[i] * result
            if prio[i] == 0:
                print(prio[i], result)
        fitness = np.exp(self.DNA_size * 2 / prio)  # 擴大fitness差異
        fitness[weather_calc] = fitness[weather_calc] * 1.065  # 若是雨天就*1.1
        return fitness, total_distance, weather_calc

    def select(self, fitness):

        idx = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        # fitness越高越可能排在越前面
        # idx 是一個隨機的數字陣列，size = self.pop_size
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(
                np.bool)   # 產出一個與DNA大小相同的True False 陣列
            keep_city = parent[~cross_points]
            # # 陣列中是 False 的 Index 會被加到新的DNA內
            swap_city = pop[i_, np.isin(
                pop[i_].ravel(), keep_city, invert=True)]
            # 在pop內把不在 keep_city 內的元素放到 swap_city
            parent[:] = np.concatenate((keep_city, swap_city))
            # 連結 產生DNA
        return parent

    def rain(self):
        if np.random.rand() < self.rain_rate:
            return True

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(
                    0, self.DNA_size)  # 0~DNA_SIZE random
                swapA, swapB = child[point], child[swap_point]  # 交換中
                child[point], child[swap_point] = swapB, swapA  # 交換囉
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)  # select high fitness DNA
        pop_copy = pop.copy()  # 複製一份
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)  # 交配ㄌ
            child = self.mutate(child)  # 突變判定
        self.pop = pop  # 在上面選到的高fitness pop 成為新的 pop


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        self.priority = np.random.randint(0, high=4, size=n_cities)
        self.terrain = np.random.randint(
            0, high=2, size=n_cities).astype(bool)
        print(self.terrain)
        plt.ion()

    def plotting(self, lx, ly, total_d, weather, solution):
        plt.cla()
        for i, txt in enumerate(self.priority):
            plt.annotate(
                txt, (self.city_position[i, 0], self.city_position[i, 1]))
            # plt.annotate(
            #     i, (self.city_position[i, 0]+0.02, self.city_position[i, 1]+0.05))
        for i, terrain in enumerate(self.terrain):
            if terrain == True:
                plt.scatter(self.city_position[i, 0],
                            self.city_position[i, 1], s=50, c='#CA7A2C')
            else:
                plt.scatter(self.city_position[i, 0],
                            self.city_position[i, 1], s=50, c='#4D5139')

        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" %
                 total_d, fontdict={'size': 20, 'color': 'red'})
        plt.text(-0.05, -0.15, "Rain=%r" %
                 weather, fontdict={'size': 15, 'color': 'red'})
        # plt.text(-0.15, -0.25, "Solution=%s" %
        #          solution, fontdict={'size': 12, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE,
        mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, rain_rate=RAIN_RATE)
temp_gen = 0
temp_fitness = 0
env = TravelSalesPerson(N_CITIES)
for generation in range(N_GENERATIONS):
    lx, ly, priority, terrain = ga.translateDNA(
        ga.pop, env.city_position, env.priority, env.terrain)
    fitness, total_distance, weather = ga.get_fitness(
        lx, ly, priority, terrain)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)  # fitness中最好的那個
    if fitness[best_idx] > temp_fitness:
        temp_gen = generation
        temp_fitness = fitness[best_idx]
    if generation % 499 == 0:
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
        print('最佳基因在', temp_gen, '代已產生')
    env.plotting(lx[best_idx], ly[best_idx],
                 total_distance[best_idx], weather[best_idx], ga.pop[best_idx])
plt.ioff()
plt.show()
