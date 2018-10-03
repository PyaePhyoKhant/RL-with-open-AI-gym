import matplotlib.pyplot as plt

constant = []
with open('epsilon_constant.txt', 'r') as f:
    for line in f:
        constant.append(float(line))

plt.plot(constant)
plt.show()

decreasing = []
with open('epsilon_decreasing.txt', 'r') as f:
    for line in f:
        decreasing.append(float(line))

plt.plot(decreasing)
plt.show()
