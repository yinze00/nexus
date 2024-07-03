import random
import numpy as np

def generate_normalized_random_float_array(n):
    # 生成随机浮点数数组
    random_floats = [random.random() for _ in range(n)]
    random_floats = np.array(random_floats)
    l2_norm = np.linalg.norm(random_floats)
    
    random_floats /= l2_norm

    # # 计算数组的最大值和最小值
    # min_value = min(random_floats)
    # max_value = max(random_floats)

    # # 归一化数组
    # normalized_array = [(x - min_value) / (max_value - min_value) for x in random_floats]

    # # 转换为 NumPy 数组
    # normalized_array_np = np.array(normalized_array)

    return random_floats

# 生成长度为 10 的已归一化随机浮点数数组
n = 64
normalized_random_array = generate_normalized_random_float_array(n)

print(np.dot(normalized_random_array,normalized_random_array))

np.set_printoptions(formatter={'float': lambda x: "{:.6f},".format(x)})
print(normalized_random_array)