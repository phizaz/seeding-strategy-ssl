import numpy as np
import matplotlib.pyplot as plt
import json

dataset = 'iris'
with open('results/badness_on_many_seeding-' + dataset + '.json') as file:
    result = json.load(file)

evaluation = result['evaluation']
acc = list(map(lambda x: x[0] / x[1], evaluation))
# rmsd = list(map(lambda x: x['rmsd'], badness))
badness_l_method_md = list(map(lambda x: x['md'], result['badness']))
badness_denclue_md = list(map(lambda x: x['md'], result['badness_denclue']))
badness_naive_md = list(map(lambda x: x['md'], result['badness_naive']))
names = result['name']

seq = list(zip(acc, badness_l_method_md, badness_denclue_md, badness_naive_md, names))
seq.sort(key=lambda x: x[0])

acc, badness_l_method_md, badness_denclue_md, badness_naive_md, names = zip(*seq)

# Example data
x = range(len(evaluation))

# Create plots with pre-defined labels.
# Alternatively, you can pass labels explicitly when calling `legend`.

fig, ax = plt.subplots()
ax.plot(x, acc, 'k--', label='accuracy')
ax.plot(x, badness_l_method_md, 'k:', label='bad l_method')
ax.plot(x, badness_denclue_md, 'k', label='bad kde')
ax.plot(x, badness_naive_md, 'k', color="red", label='bad naive')

plt.xticks(range(len(acc)), names, rotation=90)
plt.subplots_adjust(bottom=0.3)

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