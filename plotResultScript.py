from numpy import vstack
import matplotlib.pyplot as plt
import csv
from math import sqrt

rows = []

with open('results.csv', 'rt') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		rows.append(list(map(float,row[:-1])))
		

ntests = len(rows[0])

ratios = [(rows[1][i]+1e-10)/(rows[2][i]+1e-10) for i in range(ntests)]
m = sum(ratios)/ntests
print("mean = ", m)
var = sqrt(sum((r-m)**2 for r in ratios)/ntests)
print("var = ", var)


fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(xmin=rows[0][0], xmax=rows[0][-1])
fig.tight_layout()

ax.plot(rows[0], ratios)
ax.set_title('Evolution of ratio with vector length')
ax.set_xlabel('log2(vector length)')
ax.set_ylabel('ratio')
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(xmin=rows[0][0], xmax=rows[0][-1])
fig.tight_layout()

labels = ['merge sort omp', 'merge sort serial']
y = vstack([rows[2], rows[1]])
ax.stackplot(rows[0], y, labels=labels)
ax.set_title('Evolution of execution time with vector length')
ax.legend(loc='upper left')
ax.set_ylabel('Execution time (s)')
ax.set_xlabel('log2(vector length)')
plt.show()
