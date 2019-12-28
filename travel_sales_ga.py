
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

    def translateDNA(self, DNA, city_position):     # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            # city_coord 內中全部行的第一個元素填到 line_x[i, :]
            line_x[i, :] = city_coord[:, 0]
            # city_coord 內中全部行的第二個元素甜到 line_y[i. :]
            line_y[i, :] = city_coord[:, 1]
            # 結論：len(city_coord[:]) == 2,  分別儲存x, y
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        weather_calc = np.random.randint(0, 2, self.pop_size).astype(
            np.bool)  # 產生是否為雨天
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(
                np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))  # 計算距離
            # 用 diff 來xs[index+1] - xs[index]... 之類的方式來變魔術
            # 用 square 來做平方(整個陣列都平方)
            # 用 sqrt 來開根號 (畢氏定理啦)
            # 用 sum 把全部的點的距離連起來
        fitness = np.exp(self.DNA_size * 2 / total_distance)  # 擴大fitness差異
        fitness[weather_calc] = fitness[weather_calc] * 1.1  # 若是雨天就*1.1
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
        plt.ion()

    def plotting(self, lx, ly, total_d, weather):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T,
                    self.city_position[:, 1].T, s=150, c='k')
        plt.scatter(lx[0], ly[0], s=150, c='r')  # 標紅色＝起點

        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" %
                 total_d, fontdict={'size': 20, 'color': 'red'})
        plt.text(-0.10, -0.10, "Rain=%r" %
                 weather, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE,
        mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, rain_rate=RAIN_RATE)

env = TravelSalesPerson(N_CITIES)
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(
        ga.pop, env.city_position)
    fitness, total_distance, weather = ga.get_fitness(lx, ly)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)  # fitness中最好的那個
    if generation % 50 == 0:
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)

    env.plotting(lx[best_idx], ly[best_idx],
                 total_distance[best_idx], weather[best_idx])
plt.ioff()
plt.show()
