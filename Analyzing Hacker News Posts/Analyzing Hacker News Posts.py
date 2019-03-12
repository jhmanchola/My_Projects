#!/usr/bin/env python
# coding: utf-8

# # Analyzing Hacker News.com Posts
# This small project analyzes the activity in the comments section of posts taken from the [Hacker News](https://news.ycombinator.com/) website. The dataset used here however was downloaded from [kaggle.com](https://www.kaggle.com/hacker-news/hacker-news-posts). This dataset gathers posts from the year 2016, up to September 26 of that year. It organizes the data in seven columns, described as follows:
# 
# - 'title': title of the post (self explanatory)
# 
# - 'url': the url of the item being linked to
# 
# - 'num_points': the number of upvotes the post received
# 
# - 'num_comments': the number of comments the post received
# 
# - 'author': the name of the account that made the post
# 
# - 'created_at': the date and time the post was made (the time zone is Eastern Time in the US)

# In[1]:


import pandas as pd
import datetime as dt

df = pd.read_csv('HN_posts_year_to_Sep_26_2016.csv')
print(df.shape)
df.head()


# ## Choosing a path to explore:
# 
# Looking through the dataset, two post 'categories' stand out: posts that start with an 'Ask HN' string and posts that start with an 'Show HN' string. 'Ask HN' stands for 'Ask Hacker News', and this are posts from users who are looking for an answer to a specific problem they might have by asking the Hacker News community. 'Show HN' are posts from users who want to promote their work. 

# In[2]:


#Filter 'ask hn' and 'show hn' posts
#'ask hn' are questions from users, 'show hn' are projects being posted
ask_posts = df['title'].str.lower().str.startswith('ask hn')
show_posts = df['title'].str.lower().str.startswith('show hn')
ask_show_df = df[ask_posts | show_posts ]
print(ask_show_df.shape)
ask_show_df.head()


# In[3]:


#Create a new column named 'Category' which labels posts as Ask HN or Show HN
#Create dataframe of Ask HN only posts with the new column:
ask_df = ask_show_df[ask_show_df['title']
                             .str.lower()
                             .str.startswith('ask hn')].assign(Category='Ask HN')
#Create dataframe of Show HN only posts with the new column:
show_df = ask_show_df[ask_show_df['title']
                             .str.lower()
                             .str.startswith('show hn')].assign(Category='Show HN')
#Join the two dataframes
categorized_df = ask_df.append(show_df)
#Group by Category and produce mean:
mean = categorized_df.groupby('Category').mean()
print('Average comments:\n\n',mean['num_comments'])


# In[4]:


#From here on plots will be created to help visualize the results.
#Each plot is going to be numerated in the title using the following class:
class GraphTitle():
    nextNumber = 1
    def __init__(self,title): 
        self.name = 'Graph'
        self.title = title
        self.Number = GraphTitle.nextNumber
        GraphTitle.nextNumber += 1
    def getTitle(self):
        return self.name + ' ' + str(self.Number) + '. ' + self.title
#The object GraphTitle will be instatiated in the title method of each graph, 
#the getTitle method will be called and it will produce titles numbered in consecutive order
#for each graph: Graph 1., Graph 2., Graph 3., ... Graph N.


# In[5]:


#Ignore a FutureWarning that pops up
import warnings
warnings.filterwarnings('ignore')

#Plotting the averages using a point plot:
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(6.5,4))
ax = sns.pointplot(x='Category', 
                   y='num_comments',
                   color='#ff8c00',
                   data=categorized_df,scale=1.1)
sns.despine(left=True,bottom=True)
ax.axes.set_ylim(0,16)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')
ax.set_title(GraphTitle('Average Number of Comments\nfor Ask HN and Show HN').getTitle())
ax.set(xlabel='Post Category', ylabel='Avg Number of Comments')
plt.show()


# ## Reading the averages
# The average values show that there are more comments on average on asking posts than on showing posts. One reason could be that people like problem solving and by commenting in an ask post they are actually solving someone else's problem. In the showing posts, people might be solving a problem but it's harder to realize who's problem are they solving. The next step is to keep analyzing the Ask HN posts, by asking, for example, which hour of the day has the most average posts.

# In[6]:


#Find the hour of the day with the most comments for Ask HN posts:

#Convert 'created_at' column into datetime format
ask_df['created_at'] = pd.to_datetime(ask_df['created_at'],format='%m/%d/%Y %H:%M')
#Extract just the hour from the 'created_at' column, so a grouping by hour can be done
hour = ask_df['created_at'].dt.hour
#'Hour' column created to group by that column, last map() function ignores seconds
ask_df['Hour'] = pd.to_datetime(hour,format='%H').dt.time.map(lambda t: 
                                                              t.strftime('%H:%M'))
#Group by hour
avg_by_hour = ask_df.groupby('Hour').mean().sort_values(by='num_comments',ascending=False)
print('Average comments per hour (Eastern Time):\n\n',
      avg_by_hour['num_comments'].head().map('average {:,.2f} comments per post.'.format))


# ## Post Creation Hours with highest average comments
# The hour a post was created receiving the highest average of comments is 15:00 Eastern Time. By looking at this information, an Ask HN post has a higher chance to get a comment at this hour. If your time zone differs from the one used in the dataset, the following code will help:

# In[7]:


avg_by_local_hour = avg_by_hour.reset_index()

#For a user in Nairobi, Kenya, the hours with the highest average comments would differ:
#Change 'Hour' column to 'Nairobi_Time'
avg_by_local_hour.rename(columns={'Hour':'Nairobi_Time'},inplace=True)

local_UTC = 3 #Nairobi has a time of UTC+3

#Convert values in avg_by_local_hour['Hour'] to datetime objects:
avg_by_local_hour['Nairobi_Time'] = pd.to_datetime(avg_by_local_hour['Nairobi_Time'])
#Eastern time is UTC-5, so data values will get converted back to UTC by adding 5:
avg_by_local_hour['Nairobi_Time'] = avg_by_local_hour['Nairobi_Time']+dt.timedelta(hours=5)
#Now add local_UTC to the values:
avg_by_local_hour['Nairobi_Time'] = avg_by_local_hour['Nairobi_Time']+dt.timedelta(hours=local_UTC)

#Change format of 'Nairobi_Hour' to show only time and ignore seconds
Nai_hour = avg_by_local_hour['Nairobi_Time'].dt.hour
avg_by_local_hour['Nairobi_Time'] = pd.to_datetime(Nai_hour,format='%H').dt.time.map(lambda t: 
                                                              t.strftime('%H:%M'))
#Set index to be the hour:
avg_by_local_hour.set_index('Nairobi_Time',inplace=True)
print('Average coments per hour (Nairobi Time):\n\n',
      avg_by_local_hour['num_comments'].head().map('average {:,.2f} comments per post.'.format))


# ## Plot the post creation hours with highest comment average
# Back to using Eastern Time, a comparison of the numbers of comments per hour is created below:

# In[8]:


#Filter ask_df to show only the top 5 hours discovered above
ask_df_top_hrs = ask_df[ask_df['Hour'].isin(avg_by_hour['num_comments'].head().index)]
#Sort by column 'Hour'
ask_df_top_hrs = ask_df_top_hrs.sort_values(by='Hour')
#Display result
ask_df_top_hrs.head()


# In[9]:


#Plot:
#First make a list of labels for the x axis to show both Eastern Time and Nairobi Time:
labels = []
#sort times 
Etimes = avg_by_hour['num_comments'].head().sort_index()
Ntimes = avg_by_local_hour['num_comments'].head().sort_index()
#append values to list
for h in range(len(avg_by_local_hour['num_comments'].head().index)):
    labels.append('{} ET\n{} NT'.format(Etimes.index[h],
                                        Ntimes.index[h]))
#Create plot
ax = sns.catplot(x='Hour', 
                   y='num_comments',
                   hue='Hour',
                   data=ask_df_top_hrs,
                   height=5.5,aspect=1.5,jitter=0.3,palette='Dark2_r')
sns.despine(left=True,bottom=True)
ax.set_axis_labels('Hour (24hr format) ET=Eastern Time,  NT=Nairobi Time','Number of Comments')
ax.set_xticklabels(labels) #pass the labels list with the two times
plt.title(GraphTitle('Number of Ask HN Comments in top 5 Hours').getTitle())
plt.tick_params(axis='both', which='both',length=0)
plt.show()


# ## The Points System
# 
# Now that the comments activity has been explored, it would be relevant to look into the Points column, 'num_points' of the dataset. A point in the Hacker News website works as a vote. The number of points in a post equals the number of users that voted for that post. The same work done in the 'num_comments' columns will be now done in this column,. That is, determine which category, 'Show HN' or 'Ask HN, receives more points on average and depending on the answer determine at what time the comments get more points.

# In[10]:


#Use the 'categorized_df' dataframe created before, which already filtered the 'Ask HN' and 'Show HN' posts:
mean = categorized_df.groupby('Category').mean()
print('Average points:\n\n',mean['num_points'])


# In[11]:


#Create a point plot to visualize the difference:
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(6.5,4))
ax = sns.pointplot(x='Category', 
                   y='num_points',
                   color='#229954',
                   data=categorized_df,scale=1.1)
sns.despine(left=True,bottom=True)
ax.axes.set_ylim(0,20)
ax.xaxis.set_ticks_position('none') #Remove x axis ticks
ax.yaxis.set_ticks_position('none') #Remove y axis ticks
ax.yaxis.set_major_locator(ticker.MultipleLocator(4)) #Y axis intervals (step).
ax.set(xlabel='Post Category', ylabel='Avg Number of Points')
ax.set_title(GraphTitle('Average Number of Points\nfor Ask HN and Show HN').getTitle())
plt.show()


# ## 'Show HN' has a higher average of Points
# Show HN posts get more points on average than Ask HN posts, opposite dynamic than when measuring the amount of comments. The next step is to keep analyzing the Show HN posts by finding out which hour of the day has the most average points.

# In[12]:


#Find the hour of the day with the most points for Show HN posts:
#(The show_df dataframe was created at the begining of this project)

#Convert 'created_at' column into datetime format
show_df['created_at'] = pd.to_datetime(show_df['created_at'],format='%m/%d/%Y %H:%M')
#Extract just the hour from the 'created_at' column, so a grouping by hour can be done
hour = show_df['created_at'].dt.hour
#'Hour' column created to group by that column, last map() function ignores seconds
show_df['Hour'] = pd.to_datetime(hour,format='%H').dt.time.map(lambda t: 
                                                              t.strftime('%H:%M'))
#Group by hour
avgP_by_hour = show_df.groupby('Hour').mean().sort_values(by='num_points',ascending=False)
print('Average points per hour (Eastern Time):\n\n',
      avgP_by_hour['num_points'].head().map('average {:,.2f} points per post.'.format))


# ## Post Creation Hours with highest average of points
# The hour a post was created receiving the highest average of points is 12:00 Eastern Time. A Show HN post has a higher chance to get more points at this hour. A plot showing the number of points for each of the top 5 hours is  helpful to compare times. The plto is shown in the next section.

# ## Plotting the Number of Points per Hour

# In[13]:


#First create a new column to have the Nairobi Time already available:
show_df['Nairobi_Time'] = show_df['Hour']
#Convert the new column to datetime so the timedelta operation can be done
show_df['Nairobi_Time'] = pd.to_datetime(show_df['Nairobi_Time'])
#We know now that Nairobi Time equals Eastern Time + 8 hours
show_df['Nairobi_Time'] = show_df['Nairobi_Time']+dt.timedelta(hours=8)
#Format the Nairobi Time column to show just the hour and minutes as a string
Nai_hour = show_df['Nairobi_Time'].dt.hour
show_df['Nairobi_Time'] = pd.to_datetime(Nai_hour,format='%H').dt.time.map(lambda t: 
                                                              t.strftime('%H:%M'))
show_df.head(3)


# In[14]:


#Filter show_df to show only the top 5 hours discovered above
show_df_top_hrs = show_df[show_df['Hour'].isin(avgP_by_hour['num_points'].head().index)]
#Sort by column 'Hour'
show_df_top_hrs = show_df_top_hrs.sort_values(by='Hour')
#Display result
show_df_top_hrs.head(3)


# In[15]:


#Plot:
#First make a list of labels for the x axis to show both Eastern Time and Nairobi Time:
labels = []
#sort times 
Etimes = show_df_top_hrs['Hour'].unique()
Ntimes = show_df_top_hrs['Nairobi_Time'].unique()
#append values to list
for h in range(len(avg_by_local_hour['num_comments'].head().index)):
    labels.append('{} ET\n{} NT'.format(Etimes[h],
                                        Ntimes[h]))
#Create plot
ax = sns.catplot(x='Hour', 
                   y='num_points',
                   hue='Hour',
                   data=show_df_top_hrs,
                   height=5.5,aspect=1.5,jitter=0.3,palette='Dark2_r')
sns.despine(left=True,bottom=True)
ax.set_axis_labels('Hour (24hr format) ET=Eastern Time,  NT=Nairobi Time','Number of Points')
ax.set_xticklabels(labels) #pass the labels list with the two times
plt.title(GraphTitle('Number of Show HN Points in top 5 Hours').getTitle())
plt.tick_params(axis='both', which='both',length=0)
plt.show()


# Graph 4 above shows the hour of highest average points, 12:00 Eastern Time (20:00 Nairobi Time) as compared with the other top 5 hours. It can be seen that at hour 11:00 ET a single post obtained around 1,000 points.

# ## Other posts
# 
# There are other types of posts in the Hacker News website which are not labeled as Ask HN or Show HN. To make this analysis complete, an exploration of the number of comments and number of points will be performed on the other posts.

# In[16]:


other_df = df[~ask_posts | ~show_posts] #Use the ~ to indicate this variables are excluded

#Create a 'Category' column were all comments are labeled as 'Other'
other_df['Category'] = 'Other'

#Append this to the 'categorized_df' dataframe created before
new_categorized_df = categorized_df.append(other_df)
#Group by category and produce mean:
mean = new_categorized_df.groupby('Category').mean()
print('Average comments:\n\n',mean['num_comments'])


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')

#Use the melt() function to unpivot the dataframe and 
#set the columns 'num_comments' and 'num_points' as values of a column named 'Variables'
plot_df = new_categorized_df.melt('Category', 
                                  value_vars=['num_comments','num_points'],
                                  var_name='Variables',  
                                  value_name='Count')

plt.figure(figsize=(6.5,4))

ax = sns.pointplot(x='Category', 
                   y='Count',
                   hue='Variables',
                   data=plot_df,scale=1.1)
sns.despine(left=True,bottom=True)
ax.axes.set_ylim(0,18)
ax.xaxis.set_ticks_position('none') #Remove x axis ticks
ax.yaxis.set_ticks_position('none') #Remove y axis ticks
ax.set(xlabel='Post Category',ylabel='Average Number')
ax.xaxis.labelpad = 15 #Move the x axis label downward
ax.set_title(GraphTitle('Average Number of Comments\nand Points per Category').getTitle())
ax.legend(bbox_to_anchor=(1.05, 1),loc='best',title='Variables') #Move legend outside
plt.show()


# Graph 5 above shows that other posts don't receive a considerable amount of comments in average, but they do get slightly more points than the Show HN category and much more points than the Ask HN category.

# ## Conclusion
# 
# Hacker News is a website where tech and startup stories are shared. As any other posting service, it enables the interaction among its users. The stories that get the most engagement (number of comments/points) aquire more visibility hence giving more attention to the author of the post. This small project was able to establish three things:
# - What types of posts are found in Hacker News.com.
# - The average number of comments for each type of post.
# - The time of day in which a post is created may have incidence in the amount of points and comments it receives.
# 
