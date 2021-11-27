import matplotlib.pyplot as plt
import numpy as np
from numpy import isnan
import pandas as pd
from pathlib import Path
# import re
# import seaborn as sns

# from datetime import datetime
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn import preprocessing
# from sklearn.decomposition import PCA

ROOT_PATH = Path().absolute()
DATASET_PATH = ROOT_PATH / 'data' / 'XY_train.csv'
##------------read the files---------------------##
data_xy_train = pd.read_csv(DATASET_PATH)
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

def clean_nan(s: pd.Series) -> pd.Series:
    return s[~s.isna()]


def plot_hist(s: pd.Series, title: str, x_label: str, y_label: str = 'Frequency', bins=50, color='darkblue'):
    plt.hist(s, bins=bins, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.show()


if __name__ == '__main__':
    gender = clean_nan(s=data_xy_train['gender'])
    plot_hist(s=gender,
              title='Gender histogram',
              x_label='gender')
    plot_hist(s=data_xy_train['training_hours'],
              title="training_hours Histogram",
              x_label='training_hours')

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
