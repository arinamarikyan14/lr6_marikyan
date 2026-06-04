'''import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4,] [1, 4, 2, 5])
plt. ylabel('some numbers')
plt.savefig('fig')'''

from sklearn.datasets import make_blobs 
import pandas as pd
dataset, classes = make_blobs(n_samples=200, n_features=2,
centers=4, cluster_std=0.5, random_state=0)
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
print(df.head(2))
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4, n_init=10, random_state=0)
visualizer = KElbowVisualizer(model, k=(2, 12), force_model=True)
visualizer.fit(df)
visualizer.show(outpath="elbow_plot.png")