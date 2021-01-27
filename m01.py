import numpy as np
import matplotlib.pyplot as plt


# 사인곡선 그려보기

# 1 data
x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()