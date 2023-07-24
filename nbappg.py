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

# %%
y_binary = (y > 0.4).astype(int)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# %%
y_pred = logreg.predict(X_test)
threshold_line = np.array([0.4] * len(X_test))

# %%
plt.figure(figsize=(5, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted')
plt.plot(X_test, threshold_line, color='green', label='Threshold')
plt.xlabel('Points per Game')
plt.ylabel('Field Goal Percentage')
plt.title('Classification Model')


# %%
players_above_threshold = (y_test == 1).sum()
print("Number of players averaging over 25 points with a field goal percentage over 40%:", players_above_threshold)
players_under_threshold = (y_test == 0).sum()
print("Number of players averaging over 25 points with a field goal percentage under 40%:", players_under_threshold)

# %%
position_fg_percentage = nba.groupby('Pos')['FG%'].mean()
print("The average field goal percentage by position is:\n", position_fg_percentage)

# %%
#create a bar chart to show the average field goal percentage by position
plt.figure(figsize=(10, 5))
plt.bar(position_fg_percentage.index, position_fg_percentage.values, color='magenta', alpha=0.5)
plt.xlabel('Position')
plt.ylabel('Field Goal Percentage')
plt.title('Average Field Goal Percentage by Position')
plt.xticks(rotation=45)
plt.ylim(0, 0.7)
plt.plot(position_fg_percentage.index, position_fg_percentage.values, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.show()
    

# %%
print("the position with the highest average field goal percentage are:", position_fg_percentage.idxmax(), "with an average field goal percentage of:", position_fg_percentage.max())
print("the position with the lowest average field goal percentage are:", position_fg_percentage.idxmin(), "with an average field goal percentage of:", position_fg_percentage.min())


