import matplotlib.pyplot as plt
import numpy as np


thresh = ['-', 0, 0.5, 1, 2, 5]
removed = np.array([0, 18, 26, 32, 42, 58])
acc = np.array([1.86, 1.86, 1.86, 2.04, 2.51, 3.77])

plt.figure(figsize=[4, 4])
plt.plot(removed, acc)
plt.plot(removed, acc, 'x', markersize=10, markeredgewidth=3)
plt.grid('minor', alpha=0.5)
plt.text(removed[0], acc[0]-0.25, str('-'))
for i in range(1,len(thresh)):
    plt.text(removed[i], acc[i]-0.25, str(thresh[i]) + '%')
plt.xlabel('neurons removed [%]')
plt.ylabel('avg err [deg]')
plt.title('Ablation study of inactive neurons')
plt.ylim([0, 4])
plt.tight_layout()
plt.show()

