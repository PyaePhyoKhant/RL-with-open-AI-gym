import matplotlib.pyplot as plt

wo_train = []
with open('without_optimize_training.txt', 'r') as f:
    for line in f:
        wo_train.append(float(line))

w_train = []
with open('with_optimize_training.txt', 'r') as f:
    for line in f:
        w_train.append(float(line))

plt.plot(wo_train, 'b', label='without optimization')
plt.plot(w_train, 'r', label='with optimization')
plt.legend()
plt.title('Effect of optimization (training)')
plt.xlabel('episode')
plt.ylabel('score')
plt.savefig('result_training.png')
plt.show()

wo_train = []
with open('without_optimize_testing.txt', 'r') as f:
    for line in f:
        wo_train.append(float(line))

w_train = []
with open('with_optimize_testing.txt', 'r') as f:
    for line in f:
        w_train.append(float(line))

plt.plot(wo_train, 'b', label='without optimization')
plt.plot(w_train, 'r', label='with optimization')
plt.legend()
plt.title('Effect of optimization (testing)')
plt.xlabel('episode')
plt.ylabel('score')
plt.savefig('result_testing.png')
plt.show()
