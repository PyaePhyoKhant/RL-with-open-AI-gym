import matplotlib.pyplot as plt

full = []
with open('dimensions_full.txt', 'r') as f:
    for line in f:
        full.append(float(line))

cut = []
with open('dimensions_cut.txt', 'r') as f:
    for line in f:
        cut.append(float(line))

plt.plot(full, 'b', label='full')
plt.plot(cut, 'r', label='cut')
plt.legend()
plt.title('Effect of removing x and y')
plt.xlabel('episode')
plt.ylabel('score')
plt.savefig('cutting_variables.png')
plt.show()
