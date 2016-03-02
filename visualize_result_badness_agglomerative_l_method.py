import numpy as np
import matplotlib.pyplot as plt
import json

dataset = 'pendigits'
with open('results/badness_agglomerative_l_method-' + dataset + '.json') as file:
    result = json.load(file)

evaluation = result['evaluation']
acc = list(map(lambda x: x[0] / x[1], evaluation))
badness = result['badness']
rmsd = list(map(lambda x: x['rmsd'], badness))
md = list(map(lambda x: x['md'], badness))

seq = list(zip(acc, rmsd, md))
seq.sort(key=lambda x: x[0])

acc, rmsd, md = zip(*seq)

# Example data
x = range(len(evaluation))

# Create plots with pre-defined labels.
# Alternatively, you can pass labels explicitly when calling `legend`.
fig, ax = plt.subplots()
ax.plot(x, acc, 'k--', label='accuracy')
ax.plot(x, rmsd, 'k:', label='bad - rmsd')
ax.plot(x, md, 'k', label='bad - md')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()