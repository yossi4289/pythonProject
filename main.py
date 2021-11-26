import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA

##------------read the files---------------------##
data_xy_train = pd.read_csv(r'C:\Users\Owner\OneDrive\Documents\ml\XY_train.csv')
# data_x_test = pd.read_csv(r'C:\Users\shirg\Downloads\X_test.csv')
# data_json = pd.read_json(r'C:\Users\shirg\Desktop\לימוד מכונה\partOne\Jason_File.json')



# ##------------EDA---------------##
# ##--CITY--##
# plt.hist(data_xy_train['city'], bins=2000, color='darkblue', )
# plt.title("City Histogram", fontsize=20)
# plt.xlabel('city', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()
#
# #
# ##--city_development_index--##
# plt.hist(data_xy_train['city_development_index'], bins=1000, color='darkblue')
# plt.title("city_development_index Histogram", fontsize=20)
# plt.xlabel('city_development_index', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()
# #
# ##--gender--##
def create_data():
    return pd.Series(data_xy_train['gender']).values


df = create_data()
dt=np.isna(df)
#print(df.isna())
#dt=pd.Series.loc[df.isna()]
print(dt)
# print (df)
# genders = data_xy_train['gender']

# df.dropna()
print(df)

# print(df.isnull)
for gender in df:
    if gender == np.nan:
        gender = 'no dd'
print(df.isna)
df.isnull().sum()
# plt.hist(genders, bins=50, color='darkblue')
# plt.title("gender Histogram", fontsize=20)
# plt.xlabel('gender', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()

# ##--training_hours--##
plt.hist(data_xy_train['training_hours'], bins=1000, color='darkblue')
plt.title("training_hours Histogram", fontsize=20)
plt.xlabel('training_hours', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()
#
# ##--View Count Count--##
# plt.hist(y, bins=10000, color='darkblue')
# plt.title("View Count Histogram", fontsize=20)
# plt.xlabel('View Count', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()
#
# ##--Published At Histogram--##
# months_publish = list()
# time = data_xy_train['publishedAt'].values
# for date in time :
#     index1 = date.split(sep="-")
#     months_publish.append(index1[1])
# months_publish.sort()
#
# plt.hist(months_publish,bins=20, color='darkblue')
# plt.title("Published At Histogram", fontsize=20)
# plt.xlabel('Published At', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()
#
# ##--Trending Date Histogram--##
# months_trending = list()
# months = data_xy_train['trending_date'].values
# for date in months :
#     index1 = date.split(sep="-")
#     months_trending.append(index1[1])
# months_trending.sort()
#
# # plt.hist(months_trending,bins=20, color='darkblue')
# plt.title("Trending Date Histogram", fontsize=20)
# plt.xlabel('Trending Date', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()
#
# # ##--Likes and Dislikes--##
# plt.scatter(x=data_xy_train['likes'], y=data_xy_train['dislikes'])
# plt.title("Interaction between likes and dislikes", fontsize=20)
# plt.xlabel('Likes')
# plt.ylabel('Dislikes')
# plt.show()
#
# # ##--Likes and comment count--##
# plt.scatter(x=data_xy_train['comment_count'], y=data_xy_train['likes'])
# plt.title("Interaction between likes and comments", fontsize=20)
# plt.xlabel('Comment Count')
# plt.ylabel('Likes')
# plt.show()
#
#  # ##--Likes and view count--##
# plt.scatter(y= y, x=data_xy_train['likes'])
# plt.title("Interaction between likes and view count", fontsize=20)
# plt.ylabel('View Count')
# plt.xlabel('Likes')
# plt.show()
#
# ##--Dislikes and view count--##
# plt.scatter(y= y, x=data_xy_train['dislikes'])
# plt.title("Interaction between dislikes and view count", fontsize=20)
# plt.ylabel('View Count')
# plt.xlabel('Dislikes')
# plt.show()
#
# ##-------Pre Prossecing--------##
# ##--Missing Values--##
# #-empty Categories-#
# data_xy_train['cat_fixed'] = data_xy_train.apply(
#     lambda row: data_xy_train[data_xy_train['description'] == row['description']].iloc[0]['categoryId'] if np.isnan(row['categoryId']) else row['categoryId'],
#     axis=1)
# data_xy_train['categoryId'] = data_xy_train['cat_fixed']
#
# #-empty likes-#
# df_cat_id = data_xy_train[data_xy_train['categoryId'] == 10].copy()
# mean_cat_id10 = df_cat_id['likes'].mean()
# data_xy_train['likes'] = data_xy_train['likes'].fillna(mean_cat_id10)
#
# #-empty descriptions-#
# data_xy_train['description_fixed'] = data_xy_train.apply(
#     lambda row: row['description'] if row['description'] == '' else row['title'], axis=1)
# data_xy_train['description'] = data_xy_train['description_fixed']
# data_xy_train.drop('description_fixed', axis='columns', inplace=True)
# data_xy_train.drop('cat_fixed', axis='columns', inplace=True)
#
# #-empty tags-#
# x = data_xy_train.isnull()
# for key in data_xy_train.keys():
#     for i in range(len(data_xy_train[key])):
#         if x[key][i] or data_xy_train[key][i] == "[None]":
#             if key == "categoryId":
#                 if data_xy_train["title"][i] == data_xy_train["title"][i - 1] and data_xy_train["publishedAt"][i] == \
#                         data_xy_train["publishedAt"][i - 1] and not x[key][i - 1]:
#                     data_xy_train[key][i] = data_xy_train[key][i - 1]
#             if key == "tags":
#                 categoryId = data_xy_train["categoryId"][i]
#                 for item in data_json["items"]:
#                     if int(item["id"]) == int(categoryId):
#                         data_xy_train[key][i] = item["snippet"]["title"]
