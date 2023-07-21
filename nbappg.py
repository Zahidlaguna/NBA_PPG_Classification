# %%
import pandas as pd 
import numpy as np 
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %%
nba = pd.read_csv('/Users/z2271499/Downloads/nba/regularSeason.csv')
nba

# %%
nbaPPG_FG = nba.iloc[:, [1,2,10,29]]
nbaPPG_FG

# %%
nbaPPG_FG.shape

# %%
points = nbaPPG_FG['PTS'].to_numpy()
points

# %%
Fg_percentage = nbaPPG_FG['FG%'].to_numpy()
Fg_percentage

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("Points per Game")
ax1.plot(points, 'o', color='blue')
ax2.set_title("Field Goal Percentage")
ax2.plot(Fg_percentage, 'o', color='red')

plt.show()

# %%
X = points.reshape(-1, 1)
y = Fg_percentage.reshape(-1, 1)


