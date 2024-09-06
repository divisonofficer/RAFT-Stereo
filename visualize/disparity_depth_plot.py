import math
import numpy as np
import matplotlib.pyplot as plt


def plot_disparity_range(fx, T, max_disparity, INPUT_SCALE=1.0):
    def disparity_to_depth(disparity):
        baseline = math.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2)
        depth = (fx * INPUT_SCALE * baseline) / (disparity) / 1000
        return depth

    disparity_graph = np.arange(1, int(max_disparity))
    depth = disparity_to_depth(disparity_graph)

    plt.plot(disparity_graph, depth)
    plt.xlabel("Pixels")
    plt.ylabel("Meter")
