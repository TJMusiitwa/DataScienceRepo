#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline


#%%
sns.set()

#%%
tips = sns.load_dataset('tips')

#%%
sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips)

#%%
sns.catplot(x="day", y="total_bill", hue="smoker",
            kind="violin", split=True, data=tips);

#%%
from numpy.random import normal
x = normal(size=100)
plt.hist(x, bins=20)
plt.show()