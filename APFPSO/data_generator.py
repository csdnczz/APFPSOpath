import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


f = open("data.txt", 'w+')


n = 100
mu = 0
sigma = 5
M = 0.5
C = 10
rand = np.random.normal(mu,sigma,(n,))
X = np.linspace(1,n,n)


for m in range(1000):
	for c in range(1000):
		

for i in range(n):
	f.write('{0:.2f}'.format(rand[i]) + "\n")



plt.figure(figsize=(18,18))
plt.scatter(X, (M*X + C)+rand)
plt.xlabel('X-axis', fontsize=24, color='blue')
plt.ylabel('Y-axis', fontsize=24, color='blue')
plt.show()