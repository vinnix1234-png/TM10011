"""
Docstring for analysis fdfgdrhdrerh
"""
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from scipy import stats
#%%
df = pd.read_csv("datasets.csv")
nr_datasets = df["dataset"].nunique()
print("Nr of datasets:", nr_datasets)
names= df["dataset"].unique()
print("Names of datasets:", names)

status = df.groupby("dataset")[["x", "y"]].agg(["count", "mean", "var", "std"])
print(status)

for name, group in df.groupby("dataset"):
    corr = group["x"].corr(group["y"])
    conv = group["x"].cov(group["y"])
    print(name, ":", round(corr, 4),'/', round(conv, 4))

print("\nLinear regression results:")
for name, g in df.groupby("dataset"):
    lr = stats.linregress(g["x"], g["y"])
    print(f"{name}: slope={lr.slope:.4f}, intercept={lr.intercept:.4f}, r-value={lr.rvalue:.4f}")

#%%
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

seaborn.violinplot(x='dataset', y='x', data=df, ax=axes[0])
axes[0].set_title("Violin plot of x per dataset")

seaborn.violinplot(x='dataset', y='y', data=df, ax=axes[1])
axes[1].set_title("Violin plot of y per dataset")

plt.tight_layout()

g = seaborn.FacetGrid(df, col="dataset", col_wrap=4, sharex=False, sharey=False, height=5)
g.map_dataframe(seaborn.scatterplot, x="x", y="y")
g.fig.suptitle("Scatterplots per dataset", y=1.02)
plt.show()

seaborn.lmplot(data=df, x="x", y="y", col="dataset", col_wrap=4, height=7,
           scatter_kws={"s": 20}, line_kws={"linewidth": 2})
plt.suptitle("Scatterplots with regression line per dataset", y=1.02)
