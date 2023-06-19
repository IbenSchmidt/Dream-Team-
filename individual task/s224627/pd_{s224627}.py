#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Hopefully, this short tutorial can show you a lot of different commands that will help you gain the most insights into your dataset. 

# In[2]:


import pandas as pd
from src.utils import load_data_from_google_drive


# # Loading in Data

# The first step in any ML problem is identifying what format your data is in, and then loading it into whatever framework you're using. For Kaggle compeitions, a lot of data can be found in CSV files, so that's the example we're going to use. 

# We're going to be looking at a sports dataset that shows the results from NCAA basketball games from 1985 to 2016. This dataset is in a CSV file, and the function we're going to use to read in the file is called **pd.read_csv()**. This function returns a **dataframe** variable. The dataframe is the golden jewel data structure for Pandas. It is defined as "a two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns)".

# Just think of it as a table for now. 

# In[3]:


df = load_data_from_google_drive(url='https://drive.google.com/file/d/184JcLbSpArA_uq0DgAv2k892KChJVPHt/view?usp=share_link')


# In[4]:


df


# # The Basics

# Now that we have our dataframe in our variable df, let's look at what it contains. We can use the function **head()** to see the first couple rows of the dataframe (or the function **tail()** to see the last few rows).

# In[5]:


df.head()


# In[6]:


df.tail()


# We can see the dimensions of the dataframe using the the **shape** attribute

# In[7]:


df.shape


# We can also extract all the column names as a list, by using the **columns** attribute and can extract the rows with the **index** attribute

# In[8]:


df.columns.tolist()


# In order to get a better idea of the type of data that we are dealing with, we can call the **describe()** function to see statistics like mean, min, etc about each column of the dataset. 

# In[9]:


df.describe()


# Okay, so now let's looking at information that we want to extract from the dataframe. Let's say I wanted to know the max value of a certain column. The function **max()** will show you the maximum values of all columns

# In[10]:


df.max()


# Then, if you'd like to specifically get the max value for a particular column, you pass in the name of the column using the bracket indexing operator

# In[11]:


df['Wscore'].max()


# If you'd like to find the mean of the Losing teams' score. 

# In[12]:


df['Lscore'].mean()


# But what if that's not enough? Let's say we want to actually see the game(row) where this max score happened. We can call the **argmax()** function to identify the row index

# In[13]:


df['Wscore'].argmax()


# One of the most useful functions that you can call on certain columns in a dataframe is the **value_counts()** function. It shows how many times each item appears in the column. This particular command shows the number of games in each season

# In[14]:


df['Season'].value_counts()


# **Q**: How many unique seasons are there in the dataset? Use the nunique() function.

# In[15]:


df['Season'].nunique()


# **Q**: Find the team with the most wins. Use the value_counts() function on the Wteam column.

# In[16]:


df['Wteam'].value_counts()


# # Acessing Values

# Then, in order to get attributes about the game, we need to use the **iloc[]** function. Iloc is definitely one of the more important functions. The main idea is that you want to use it whenever you have the integer index of a certain row that you want to access. As per Pandas documentation, iloc is an "integer-location based indexing for selection by position."

# In[17]:


df.iloc[[df['Wscore'].argmax()]]


# Let's take this a step further. Let's say you want to know the game with the highest scoring winning team (this is what we just calculated), but you then want to know how many points the losing team scored. 

# In[18]:


df.iloc[[df['Wscore'].argmax()]]['Lscore']


# When you see data displayed in the above format, you're dealing with a Pandas **Series** object, not a dataframe object.

# In[19]:


type(df.iloc[[df['Wscore'].argmax()]]['Lscore'])


# In[20]:


type(df.iloc[[df['Wscore'].argmax()]])


# The following is a summary of the 3 data structures in Pandas (Haven't ever really used Panels yet)
# 
# ![](DataStructures.png)

# When you want to access values in a Series, you'll want to just treat the Series like a Python dictionary, so you'd access the value according to its key (which is normally an integer index)

# In[21]:


df.iloc[[df['Wscore'].argmax()]]['Lscore'][24970]


# The other really important function in Pandas is the **loc** function. Contrary to iloc, which is an integer based indexing, loc is a "Purely label-location based indexer for selection by label". Since all the games are ordered from 0 to 145288, iloc and loc are going to be pretty interchangable in this type of dataset

# In[22]:


df.iloc[:3]


# In[23]:


df.loc[:3]


# Notice the slight difference in that iloc is exclusive of the second number, while loc is inclusive. 

# Below is an example of how you can use loc to acheive the same task as we did previously with iloc

# In[24]:


df.loc[df['Wscore'].argmax(), 'Lscore']


# A faster version uses the **at()** function. At() is really useful wheneever you know the row label and the column label of the particular value that you want to get. 

# In[25]:


df.at[df['Wscore'].argmax(), 'Lscore']


# If you'd like to see more discussion on how loc and iloc are different, check out this great Stack Overflow post: http://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation. Just remember that **iloc looks at position** and **loc looks at labels**. Loc becomes very important when your row labels aren't integers. 

# # Sorting

# Let's say that we want to sort the dataframe in increasing order for the scores of the losing team

# In[26]:


df.sort_values('Lscore').head()


# **Q**: Make three dataframes that are sorted by season, winning team, and winning score respectively. Then, Using iloc, select the rows from index 100 to 200 and the columns for season, winning team, and winning score, respectively. 

# In[27]:


df.sort_values('Season')


# In[28]:


df.sort_values('Wteam')


# In[29]:


df.sort_values('Wscore')


# In[30]:


df.iloc[100:201][['Season','Wteam','Wscore']]


# **Q**: From these three subsets you obtained above, find the season and winning team for the game with the highest winning score.

# In[31]:


max_score=df['Wscore'].idxmax()


# In[32]:


df.loc[max_score, 'Season']


# In[33]:


df.loc[max_score, 'Wteam']


# # Filtering Rows Conditionally

# Now, let's say we want to find all of the rows that satisy a particular condition. For example, I want to find all of the games where the winning team scored more than 150 points. The idea behind this command is you want to access the column 'Wscore' of the dataframe df (df['Wscore']), find which entries are above 150 (df['Wscore'] > 150), and then returns only those specific rows in a dataframe format (df[df['Wscore'] > 150]).

# In[34]:


df[df['Wscore'] > 150]


# This also works if you have multiple conditions. Let's say we want to find out when the winning team scores more than 150 points and when the losing team scores below 100. 

# In[35]:


df[(df['Wscore'] > 150) & (df['Lscore'] < 100)]


# **Q**: Create a new column in the DataFrame called 'ScoreDifference' which is the absolute difference between the winning score and the losing score. Filter the DataFrame to only include games where the 'ScoreDifference' is greater than the average 'ScoreDifference' for all games.

# In[36]:


df['ScoreDifference'] = abs(df['Wscore'] - df['Lscore'])
df[df['ScoreDifference']>df['Lscore'].mean()]


# **Q**: From this filtered DataFrame, find the season and teams involved in the game with the highest 'ScoreDifference'.

# In[37]:


max_diff = df['ScoreDifference'].idxmax()


# In[38]:


df.loc[max_diff, 'Season']


# In[39]:


df.loc[max_diff, 'Wteam']


# In[40]:


df.loc[max_diff, 'Lteam']


# # Grouping

# Another important function in Pandas is **groupby()**. This is a function that allows you to group entries by certain attributes (e.g Grouping entries by Wteam number) and then perform operations on them. The following function groups all the entries (games) with the same Wteam number and finds the mean for each group. 

# In[41]:


df.groupby('Wteam')['Wscore'].mean().head()


# This next command groups all the games with the same Wteam number and finds where how many times that specific team won at home, on the road, or at a neutral site

# In[42]:


df.groupby('Wteam')['Wloc'].value_counts().head(9)


# Each dataframe has a **values** attribute which is useful because it basically displays your dataframe in a numpy array style format

# In[43]:


df.values


# Now, you can simply just access elements like you would in an array. 

# In[44]:


df.values[0][0]


# **Q**: Group the DataFrame by season and find the average winning score for each season.

# In[45]:


df.groupby('Season').mean('Wscore')


# **Q**: Group the DataFrame by winning team and find the maximum winning score for each team across all seasons.

# In[46]:


df.groupby('Wteam').max('Wscore')


# **Q**: Group the DataFrame by both season and winning team. Find the team with the highest average winning score for each season.

# In[58]:


# Group the dataframe by season and winning team
grouped = df.groupby(['Season', 'Wteam'])

# Calculate the average of the winning score for each team every season
average_scores = grouped['Wscore'].mean()

# Find the team with the highest average winning score for each season
teams_with_highest_scores = average_scores.groupby('Season').idxmax()

# Print the results
for season, team_idx in teams_with_highest_scores.items():
    team = team_idx[1]
    highest_average_score = average_scores.loc[team_idx]
    print(f"Season {season}: Team {team} has the highest average winning score: {highest_average_score}")


# **Q**: Create a new DataFrame that counts the number of wins for each team in each season. This will involve grouping by both season and winning team, and then using the count() function.

# In[48]:


# Create a new datafram that counts the amount of wins for each team each season
win_counts = df.groupby(['Season', 'Wteam']).count()['Wscore'].reset_index()

# Rename the coloumn 'Wsccore' to 'WinCount'
win_counts = win_counts.rename(columns={'Wscore': 'WinCount'})

# Print the results
print(win_counts)


# **Q**: For each season, find the team with the most wins. This will involve creating a DataFrame similar to the one in task 5, and then using the idxmax() function for each season.

# In[57]:


season_wins = df.groupby(['Season', 'Wteam']).size().reset_index(name='Wscore')
team_with_most_wins = season_wins.groupby('Season')['Wscore'].idxmax()
team_most_wins_per_season = season_wins.loc[team_with_most_wins]

for index, row in team_most_wins_per_season.iterrows():
    season = row['Season']
    team = row['Wteam']
    wins = row['Wscore']
    print(f"Season: {season}, Team with most wins: {team}, Wins: {wins}")


# **Q**: Group the DataFrame by losing team and find the average losing score for each team across all seasons. Compare this with the average winning score for each team from task 3. Are there teams that have a higher average losing score than winning score?

# In[51]:


# Group the dataframe by losing team and find the average losing score
losing_avg_scores = df.groupby('Lteam')['Lscore'].mean()

# Print the results 
print(losing_avg_scores)


# # Dataframe Iteration

# In order to iterate through dataframes, we can use the **iterrows()** function. Below is an example of what the first two rows look like. Each row in iterrows is a Series object

# In[64]:


for index, row in df.iterrows():
    print(row)
    if index == 1:
        break


# **Q**: Create a new column 'HighScoringGame' that is 'Yes' if the winning score is greater than 100 and 'No' otherwise. This will require iterating over the rows of the DataFrame and checking the value of the winning score for each row.

# In[52]:


# Create a empty list to save the values for the new coloumn 
high_scoring_game = []

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    if row['Wscore'] > 100:
        high_scoring_game.append('Yes')
    else:
        high_scoring_game.append('No')

# Add the new coloumn to the dataframe
df['HighScoringGame'] = high_scoring_game

# Print the dataframe with the newly added coloumn 
print(df)


# **Q**: Calculate the total number of games played by each team, whether they won or lost. This will require iterating over the rows of the DataFrame and updating a dictionary that keeps track of the number of games for each team.

# In[53]:


# Create a empty dictionary to keep track af the amount of games for each team 
team_games = {}

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    # Update the amount of games for the team that won
    if row['Wteam'] in team_games:
        team_games[row['Wteam']] += 1
    else:
        team_games[row['Wteam']] = 1

    # Update the amount af gaems for the team that lost
    if row['Lteam'] in team_games:
        team_games[row['Lteam']] += 1
    else:
        team_games[row['Lteam']] = 1

# Print the amount of games for each team 
for team, games in team_games.items():
    print(f"{team}: {games} games")


# **Q**: For each season, find the game with the highest score difference (winning score - losing score). This will require iterating over the rows of the DataFrame, keeping track of the highest score difference for each season, and updating it if a game with a higher score difference is found.

# In[54]:


# Create a empty dictionary to keep track of the size of the scoredifference for each season 
max_score_diff_per_season = {}

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    season = row['Season']
    score_diff = row['Wscore'] - row['Lscore']
    
    # Update the size of the scoredifference for the current season
    if season in max_score_diff_per_season:
        if score_diff > max_score_diff_per_season[season]:
            max_score_diff_per_season[season] = score_diff
    else:
        max_score_diff_per_season[season] = score_diff

# Print the largest scoredifference for each season
for season, score_diff in max_score_diff_per_season.items():
    print(f"Season {season}: Max score difference = {score_diff}")


# Remember, iterating over a DataFrame should generally be avoided if a vectorized operation can be used instead, as vectorized operations are usually much faster. However, these tasks are designed to give practice with DataFrame iteration for cases where it might be necessary.

# Vectorized Operation Example: Create a new column 'HighScoringGame' in the DataFrame using a vectorized operation. This column should contain 'Yes' if the winning score is greater than 100 and 'No' otherwise. Use the np.where function from the numpy library for this task.

# In[69]:


import numpy as np
df['HighScoringGame'] = np.where(df['Wscore'] > 100, 'Yes', 'No')


# **Q**: Vectorized Operation: Calculate the total number of games played by each team, whether they won or lost. Instead of iterating over the DataFrame, use the value_counts() function on the winning team and losing team columns separately, and then add the two Series together.

# In[55]:


# Calculate the amount of games for the winning and losing team 
winning_games = df['Wteam'].value_counts()
losing_games = df['Lteam'].value_counts()

# Add the two series together to get the collected amount of games for each team
total_games = winning_games.add(losing_games, fill_value=0)

# Print the amount of games for each season
print(total_games)


# **Q**: For each season, find the game with the highest score difference (winning score - losing score). Instead of iterating over the DataFrame, create a new column 'ScoreDifference' using vectorized subtraction, then use the groupby() function and idxmax() function to find the game with the highest score difference for each season.

# In[56]:


# Create a new coloumn called 'ScoreDifference' by subtracting the losing score from the winning score
df['ScoreDifference'] = df['Wscore'] - df['Lscore']

# Find the index for each row with the max scoredifference for each season
max_score_diff_indices = df.groupby('Season')['ScoreDifference'].idxmax()

# Get the corresponding row with the max scoredifference
games_with_max_score_diff = df.loc[max_score_diff_indices]

# Print the results
print(games_with_max_score_diff)


# # Extracting Rows and Columns

# The bracket indexing operator is one way to extract certain columns from a dataframe.

# In[38]:


df[['Wscore', 'Lscore']].head()


# Notice that you can acheive the same result by using the loc function. Loc is a veryyyy versatile function that can help you in a lot of accessing and extracting tasks. 

# In[39]:


df.loc[:, ['Wscore', 'Lscore']].head()


# Note the difference is the return types when you use brackets and when you use double brackets. 

# In[40]:


type(df['Wscore'])


# In[41]:


type(df[['Wscore']])


# You've seen before that you can access columns through df['col name']. You can access rows by using slicing operations. 

# In[42]:


df[0:3]


# Here's an equivalent using iloc

# In[43]:


df.iloc[0:3,:]


# # Data Cleaning

# One of the big jobs of doing well in Kaggle competitions is that of data cleaning. A lot of times, the CSV file you're given (especially like in the Titanic dataset), you'll have a lot of missing values in the dataset, which you have to identify. The following **isnull** function will figure out if there are any missing values in the dataframe, and will then sum up the total for each column. In this case, we have a pretty clean dataset.

# In[44]:


df.isnull().sum()


# If you do end up having missing values in your datasets, be sure to get familiar with these two functions. 
# * **dropna()** - This function allows you to drop all(or some) of the rows that have missing values. 
# * **fillna()** - This function allows you replace the rows that have missing values with the value that you pass in.

# # Other Useful Functions

# * **drop()** - This function removes the column or row that you pass in (You also have the specify the axis). 
# * **agg()** - The aggregate function lets you compute summary statistics about each group
# * **apply()** - Lets you apply a specific function to any/all elements in a Dataframe or Series
# * **get_dummies()** - Helpful for turning categorical data into one hot vectors.
# * **drop_duplicates()** - Lets you remove identical rows

# # Lots of Other Great Resources

# Pandas has been around for a while and there are a lot of other good resources if you're still interested on getting the most out of this library. 
# * http://pandas.pydata.org/pandas-docs/stable/10min.html
# * https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python
# * http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/
# * https://www.dataquest.io/blog/pandas-python-tutorial/
# * https://drive.google.com/file/d/0ByIrJAE4KMTtTUtiVExiUGVkRkE/view
# * https://www.youtube.com/playlist?list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y
