import matplotlib.pyplot as plt

plt.plot([0, 4], [0, 0], linewidth=4, label='x2 = 0', alpha=0.5)

x2_2 = list(map(lambda x: 8 - 2 * x, [3, 4]))
plt.plot([3, 4], x2_2, linewidth=4, label='2x_1 + x_2 = 8', alpha=0.5)

x2_3 = list(map(lambda x: (7 - x) / 2, [1, 3]))
plt.plot([1, 3], x2_3, linewidth=4, label='x_1 + 2x_2 = 7', alpha=0.5)

plt.plot([0, 0], [0, 3], linewidth=4, label='x1 = 0', alpha=0.5)

x2_1 = list(map(lambda x: 3, [0, 1]))
plt.plot([0, 1], x2_1, linewidth=4, label='x_2 = 3', alpha=0.5)

for idx, point in enumerate([[0, 0], [4, 0], [3, 2]]):
    plt.scatter([point[0]], point[1], s=100, label=f'Point #{idx + 1}', zorder=10)

plt.title('Constraints and steps')

plt.xlabel('X1')
plt.ylabel('X2', rotation=0)

plt.xticks(range(-1, 6))
plt.yticks(range(-1, 6))

plt.xlim(-1, 5)
plt.ylim(-1, 5)

plt.grid()
plt.legend()

plt.savefig('Lab3_question5.png', dpi=700)

plt.show()
