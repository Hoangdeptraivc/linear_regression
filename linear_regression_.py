import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu
data = pd.read_csv(r'C:\Users\vanho\PycharmProjects\pythonProject3\dataset\data1.csv')

# Chuẩn hóa dữ liệu
data['height'] = (data['height'] - data['height'].mean()) / data['height'].std()
data['weight'] = (data['weight'] - data['weight'].mean()) / data['weight'].std()

def loss_function(points, a, b):
    total_error = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].height
        y = points.iloc[i].weight
        total_error += (y - (a * x + b)) ** 2
    return total_error / n

def linear_regression(points, L, a, b):
    a_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].height
        y = points.iloc[i].weight
        a_gradient += (-2 / n) * x * (y - (a * x + b))
        b_gradient += (-2 / n) * (y - (a * x + b))

    a_new = a - L * a_gradient
    b_new = b - L * b_gradient
    return a_new, b_new

# Khởi tạo tham số
a_new = 0
b_new = 0
solanlap = 1000
L = 0.01  # Giảm learning rate

# Thực hiện gradient descent
for i in range(solanlap):
    if i % 50 == 0:
        print(f"solanlap: {i}, loss: {loss_function(data, a_new, b_new)}")
    a_new, b_new = linear_regression(data, L, a_new, b_new)

print(f"a_new: {a_new}, b_new: {b_new}")

# Vẽ biểu đồ
plt.scatter(data.height, data.weight, color='black')
plt.plot(data.height, a_new * data.height + b_new, c='blue')
plt.xlabel('Height (standardized)')
plt.ylabel('Weight (standardized)')
plt.show()
