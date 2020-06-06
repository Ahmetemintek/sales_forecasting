#!/usr/bin/env python
# coding: utf-8

# # PREDİCT FUTURE SALES
# #### 1- General Information About Data
# #### 2- Checking For Null Values
# #### 3- Overwiev About Outliers
# #### 4- Exploratory Data Analysis
# #### 5- Clustering The Data
# #### 6- Applying XGBoost
# #### 7- Applying RNN Model
# #### 8- Results

# ### 1- General Information About Data

# In this project we will work with a time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. The aim is to forecast the total amount of products sold in every shop for the test set. We have 5 different datasets:
# - Training Data
# - Test Data
# - Items Data
# - Item Categories Data
# - Shops Names Data

# In[1]:


# Importing the usefull libraries
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from scipy.stats.mstats import winsorize
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")
title_font= {"family": "arial", "weight": "bold", "color": "darkred", "size":13}
label_font= {"family": "arial", "weight": "bold", "color": "darkblue", "size":10}


# In[2]:


df_train= pd.read_csv("Desktop/Bootcamp/bitirme_proje/sales_train.csv") #This is the train data
df_item= pd.read_csv("Desktop/Bootcamp/bitirme_proje/items.csv") #This dataset includes names of items.
df_category= pd.read_csv("Desktop/Bootcamp/bitirme_proje/item_categories.csv") #This dataset includes category names of items.
df_shop= pd.read_csv("Desktop/Bootcamp/bitirme_proje/shops.csv")  #This dataset include name of shops. 
df_test= pd.read_csv("Desktop/Bootcamp/bitirme_proje/test.csv")


# In[3]:


df_train.info()


# In[4]:


df_item.info()


# In[5]:


df_category.info()


# In[6]:


df_shop.info()


# ### 2- Checking For Null Values

# In[7]:


df_train.isnull().sum()*100/df_train.shape[0]


# In[8]:


df_item.isnull().sum()*100/df_item.shape[0]


# In[9]:


df_category.isnull().sum()*100/df_category.shape[0]


# In[10]:


for col in df_category.columns:
    print(df_category[col].nunique())    # There are 84 different category value each columns.


# In[11]:


df_shop.isnull().sum()*100/df_shop.shape[0]


# In[12]:


for col in df_shop.columns:
    print(df_shop[col].nunique())  # There are 60 different value each columns.


# Nice! There are not appear any null value.

# ### 3- Overwiev About Outliers

# 3.1- Firstly we will check all dataframes with boxplots.

# In[13]:


# Checking for df1
plt.figure(figsize=(14,7))
col_names= ['date_block_num', 'shop_id', 'item_id', 'item_price',
            'item_cnt_day']

for i in range(5):
    plt.subplot(2,3,i+1)
    ax= sns.boxplot(x= df_train[col_names[i]], linewidth=2.5)
    plt.title(col_names[i], fontdict=title_font)    #There are some outliers for item_price and item_cnt_day columns. 
plt.show()


# In[14]:


# Checking for df2
plt.figure(figsize=(11,6))
col_names= ['item_id', 'item_category_id']

for i in range(2):
    plt.subplot(1,2,i+1)
    ax= sns.boxplot(x= df_item[col_names[i]], linewidth=2.5)
    plt.title(col_names[i], fontdict=title_font)  
plt.show()


# In[15]:


plt.figure(figsize=(10,6))
ax= sns.boxplot(x=df_category["item_category_id"], linewidth=2.5)
plt.title("item_category_id", fontdict=title_font)
plt.show()


# In[16]:


plt.figure(figsize=(10,6))
ax= sns.boxplot(x=df_shop["shop_id"], linewidth=2.5)
plt.title("shop_id", fontdict=title_font)
plt.show()


# 3.2- We will get rid of outliers.

# Trying winsorization to get rid of outliers.

# In[17]:


df_train["win_item_price"]= winsorize(df_train["item_price"], (0,0.10))
df_train["win_item_cnt_day"]= winsorize(df_train["item_cnt_day"], (0,0.10))


# In[18]:


plt.subplot(121)
ax= sns.boxplot(x=df_train["win_item_price"], linewidth=2.5)
plt.title("win_item_price", fontdict=title_font)

plt.subplot(122)
ax= sns.boxplot(x=df_train["win_item_cnt_day"], linewidth=2.5)
plt.title("win_item_cnt_day", fontdict=title_font)
plt.show()  #Nice! There is not appears any outliers by winsorization in win_item_price. 


# Trying logarithmic transformation.

# In[19]:


df_train["log_item_cnt_day"]= np.log(df_train["item_cnt_day"])


# In[20]:


ax= sns.boxplot(x=df_train["log_item_cnt_day"])
plt.title("log_item_cnt_day", fontdict=title_font)
plt.show()  # We tried log transformation but we can't get rid of outlier on item_cnt_day. 


# In[21]:


df_train= df_train.drop(df_train[df_train["item_price"]>3000].index)  #We droped the outliers
df_train= df_train.drop(df_train[df_train["item_cnt_day"]>2000].index)


# In[22]:


df_train[(df_train["item_price"]>3000)]


# In[23]:


df_train[(df_train["item_cnt_day"]>2000)]


# In[24]:


del df_train["win_item_price"]  #Modelde bunu kullanabilirim.
del df_train["win_item_cnt_day"]
del df_train["log_item_cnt_day"]


# ### 4- Exploratory Data Analysis

# #### 4.1- Firts of all we will combine and prepare all datas. 

# In[25]:


df_train= pd.merge(df_train, df_item, on="item_id", how="inner")
df_train= pd.merge(df_train, df_category, on="item_category_id", how="inner")
df_train= pd.merge(df_train, df_shop, on="shop_id", how="inner")
df_train.head()


# In[26]:


df_train.describe()


# Our values are in Russian. We will translate into English.

# In[27]:


dict_categories = ['Cinema - DVD', 'PC Games - Standard Editions',
                    'Music - Local Production CD', 'Games - PS3', 'Cinema - Blu-Ray',
                    'Games - XBOX 360', 'PC Games - Additional Editions', 'Games - PS4',
                    'Gifts - Stuffed Toys', 'Gifts - Board Games (Compact)',
                    'Gifts - Figures', 'Cinema - Blu-Ray 3D',
                    'Programs - Home and Office', 'Gifts - Development',
                    'Gifts - Board Games', 'Gifts - Souvenirs (on the hinge)',
                    'Cinema - Collection', 'Music - MP3', 'Games - PSP',
                    'Gifts - Bags, Albums, Mouse Pads', 'Gifts - Souvenirs',
                    'Books - Audiobooks', 'Gifts - Gadgets, robots, sports',
                    'Accessories - PS4', 'Games - PSVita',
                    'Books - Methodical materials 1C', 'Payment cards - PSN',
                    'PC Games - Digit', 'Games - Game Accessories', 'Accessories - XBOX 360',
                    'Accessories - PS3', 'Games - XBOX ONE', 'Music - Vinyl',
                    'Programs - 1C: Enterprise 8', 'PC Games - Collectible Editions',
                    'Gifts - Attributes', 'Service Tools',
                    'Music - branded production CD', 'Payment cards - Live!',
                    'Game consoles - PS4', 'Accessories - PSVita', 'Batteries',
                    'Music - Music Video', 'Game Consoles - PS3',
                    'Books - Comics, Manga', 'Game Consoles - XBOX 360',
                    'Books - Audiobooks 1C', 'Books - Digit',
                    'Payment cards (Cinema, Music, Games)', 'Gifts - Cards, stickers',
                    'Accessories - XBOX ONE', 'Pure media (piece)',
                    'Programs - Home and Office (Digital)', 'Programs - Educational',
                    'Game consoles - PSVita', 'Books - Artbooks, encyclopedias',
                    'Programs - Educational (Digit)', 'Accessories - PSP',
                    'Gaming consoles - XBOX ONE', 'Delivery of goods',
                    'Payment Cards - Live! (Figure) ',' Tickets (Figure) ',
                    'Music - Gift Edition', 'Service Tools - Tickets',
                    'Net media (spire)', 'Cinema - Blu-Ray 4K', 'Game consoles - PSP',
                    'Game Consoles - Others', 'Books - Audiobooks (Figure)',
                    'Gifts - Certificates, Services', 'Android Games - Digit',
                    'Programs - MAC (Digit)', 'Payment Cards - Windows (Digit)',
                    'Books - Business Literature', 'Games - PS2', 'MAC Games - Digit',
                    'Books - Computer Literature', 'Books - Travel Guides',
                    'PC - Headsets / Headphones', 'Books - Fiction',
                    'Books - Cards', 'Accessories - PS2', 'Game consoles - PS2',
                    'Books - Cognitive literature']
dict_shops = ['Moscow Shopping Center "Semenovskiy"', 
              'Moscow TRK "Atrium"', 
              "Khimki Shopping Center",
              'Moscow TC "MEGA Teply Stan" II', 
              'Yakutsk Ordzhonikidze, 56',
              'St. Petersburg TC "Nevsky Center"', 
              'Moscow TC "MEGA Belaya Dacha II"',
              'Voronezh (Plekhanovskaya, 13)', 
              'Yakutsk Shopping Center "Central"',
              'Chekhov SEC "Carnival"', 
              'Sergiev Posad TC "7Ya"',
              'Tyumen TC "Goodwin"',
              'Kursk TC "Pushkinsky"', 
              'Kaluga SEC "XXI Century"',
              'N.Novgorod Science and entertainment complex "Fantastic"',
              'Moscow MTRC "Afi Mall"',
              'Voronezh SEC "Maksimir"', 'Surgut SEC "City Mall"',
              'Moscow Shopping Center "Areal" (Belyaevo)', 'Krasnoyarsk Shopping Center "June"',
              'Moscow TK "Budenovsky" (pav.K7)', 'Ufa "Family" 2',
              'Kolomna Shopping Center "Rio"', 'Moscow Shopping Center "Perlovsky"',
              'Moscow Shopping Center "New Century" (Novokosino)', 'Omsk Shopping Center "Mega"',
              'Moscow Shop C21', 'Tyumen Shopping Center "Green Coast"',
              'Ufa TC "Central"', 'Yaroslavl shopping center "Altair"',
              'RostovNaDonu "Mega" Shopping Center', '"Novosibirsk Mega "Shopping Center',
              'Samara Shopping Center "Melody"', 'St. Petersburg TC "Sennaya"',
              "Volzhsky Shopping Center 'Volga Mall' ",
              'Vologda Mall "Marmelad"', 'Kazan TC "ParkHouse" II',
              'Samara Shopping Center ParkHouse', '1C-Online Digital Warehouse',
              'Online store of emergencies', 'Adygea Shopping Center "Mega"',
              'Balashikha shopping center "October-Kinomir"' , 'Krasnoyarsk Shopping center "Vzletka Plaza" ',
              'Tomsk SEC "Emerald City"', 'Zhukovsky st. Chkalov 39m? ',
              'Kazan Shopping Center "Behetle"', 'Tyumen SEC "Crystal"',
              'RostovNaDonu TRK "Megacenter Horizon"',
              '! Yakutsk Ordzhonikidze, 56 fran', 'Moscow TC "Silver House"',
              'Moscow TK "Budenovsky" (pav.A2)', "N.Novgorod SEC 'RIO' ",
              '! Yakutsk TTS "Central" fran', 'Mytishchi TRK "XL-3"',
              'RostovNaDonu TRK "Megatsentr Horizon" Ostrovnoy', 'Exit Trade',
              'Voronezh SEC City-Park "Grad"', "Moscow 'Sale'",
              'Zhukovsky st. Chkalov 39m² ',' Novosibirsk Shopping Mall "Gallery Novosibirsk"']


# In[28]:


df_train["item_category_name"]= df_train["item_category_name"].map(
         dict(zip(df_train["item_category_name"].value_counts().index, dict_categories)))
df_train["shop_name"]= df_train["shop_name"].map(
         dict(zip(df_train["shop_name"].value_counts().index, dict_shops)))


# We will prepare the date column to pandas datetime.

# In[29]:


df_train["date"]= pd.to_datetime(df_train["date"], format="%d.%m.%Y")
df_train["month"]= df_train["date"].dt.month
df_train["day"]= df_train["date"].dt.day
df_train["year"]= df_train["date"].dt.year
df_train.head(20)  #Seems nice!


# #### 4.2- Explore Of Data

# We examine correlation between features. 

# In[30]:


correlation_matrix= df_train.corr()
plt.figure(figsize=(20,8))
sns.heatmap(correlation_matrix, square=True, annot=True, vmin=0, vmax=1, linewidth=0.5)
plt.title("Correlation Matrix", fontdict=title_font)
plt.show()


# - item_price, total_profit and item_cnt_day has more correlate.

# Let's do comparison between the most valuable shops and most profitable shops. 

# In[31]:


df_train["total_revenue"]= df_train["item_price"]*df_train["item_cnt_day"]


# In[32]:


plt.figure(figsize=(20,7))
plt.subplot(211)
sns.barplot(data=df_train, x="shop_name", y="total_revenue")
plt.xticks(rotation=90)
plt.title("Total Revenue Per Shop", fontdict=title_font)
plt.xlabel("Shop ID", fontdict=label_font)
plt.ylabel("Total Revenue", fontdict=label_font)

plt.subplot(212)
sns.barplot(data=df_train, x="shop_name", y="item_price")
plt.xticks(rotation=90)
plt.title("Total Item Prices Per Shops", fontdict=title_font)
plt.xlabel("Shop ID", fontdict=label_font)
plt.ylabel("Total Item Prices", fontdict=label_font)
plt.show()


# In[33]:


df_price= pd.DataFrame(df_train.groupby("shop_name")["item_price"].sum().nlargest(10))
df_revenue= pd.DataFrame(df_train.groupby("shop_name")["total_revenue"].sum().nlargest(10))

plt.figure(figsize=(15,9))
sns.barplot(df_price.index, df_price["item_price"].nlargest(10), palette="Blues_d")
plt.xticks(rotation=90)
plt.title("İtem Prices Per Shop (Top 10 Shop)", fontdict=title_font)
plt.xlabel("Shop Name", fontdict=label_font)
plt.ylabel("Total Price (Million)", fontdict=label_font)
plt.show()


# In[34]:


plt.figure(figsize=(15,9))
sns.barplot(df_revenue.index, df_revenue["total_revenue"].nlargest(10), palette="Blues_d")
plt.xticks(rotation=90)
plt.title("Total Revenue Per Shops (Top 10 Shop)", fontdict=title_font)
plt.xlabel("Shop Name", fontdict=label_font)
plt.ylabel("Total Revenue", fontdict=label_font)
plt.show()


# When we look plots, there is some differences between total item valuability and total profit. 

# Now, we will try to examine sales by the time.

# In[35]:


plt.figure(figsize=(17,11))
plt.subplot(211)
sns.lineplot(data=df_train, x="year", y="item_cnt_day")
plt.title("Sales In Shop By Years", fontdict=title_font)
plt.xlabel("Year", fontdict=label_font)
plt.ylabel("Sales", fontdict=label_font)

plt.subplot(212)
sns.lineplot(data=df_train, x="year", y="total_revenue")
plt.title("Revenues In Shops By Year", fontdict=title_font)
plt.xlabel("Year", fontdict=label_font)
plt.ylabel("Revenue", fontdict=label_font)
plt.show()


# - Sales increased until 2014, however after 2014 that was decreased.
# - Profits increased all the time. 

# In[36]:


plt.figure(figsize=(17,11))
plt.subplot(211)
sns.lineplot(data=df_train, x="month", y="item_cnt_day")
plt.title("Sales In Shops By Months", fontdict=title_font)
plt.xlabel("Months", fontdict=label_font)
plt.ylabel("Sales", fontdict=label_font)

plt.subplot(212)
sns.lineplot(data=df_train, x="month", y="total_revenue")
plt.title("Revenues In Shops By Month", fontdict=title_font)
plt.xlabel("Month", fontdict=label_font)
plt.ylabel("Revenue", fontdict=label_font)
plt.show()


# The highest sales in year are in 9, 10, 11. months. But, the most total profit are in 11. and 12. month.

# Let's examine Which products are sold the most.

# In[37]:


plt.figure(figsize=(13,8))
sns.barplot(data=df_train, x="item_category_name", y="item_cnt_day", )
plt.xticks(rotation=90, color="black")  
plt.title("Daily Item Sales Per Item Category", fontdict=title_font)
plt.xlabel("Item Category ID", fontdict=label_font)
plt.ylabel("Item Sales Per Day", fontdict=label_font)
plt.show()


# Some items come to the fore in sales.

# In[38]:


df_cnt= pd.DataFrame(df_train.groupby("item_category_name")["item_cnt_day"].sum().nlargest(5))
df_rvn= pd.DataFrame(df_train.groupby("item_category_name")["total_revenue"].sum().nlargest(5))

plt.figure(figsize=(18,10))
plt.subplot(211)
sns.barplot(df_cnt.index, df_cnt["item_cnt_day"].nlargest(5), palette="Blues_d")
plt.title("Daily Sales Per Top 5 Categories", fontdict=title_font)
plt.xlabel("Category Name", fontdict=label_font)
plt.ylabel("Total Daily Sales", fontdict=label_font)

plt.subplot(212)
sns.barplot(df_rvn.index, df_rvn["total_revenue"], palette="Blues_d")
plt.title("Total Revenue Per Top 5 Categories", fontdict=title_font)
plt.xlabel("Category Name", fontdict=label_font)
plt.ylabel("Revenue", fontdict=label_font)
plt.show()


# There are some differences between top category sales and top category profits. 

# In[39]:


df_train["year-month"]= pd.to_datetime(df_train["date"]).dt.to_period("M")
df_train.head(20)   #We created new column as "year-month" in df_train for making easy the grouping.


# ### 5- Clustering The Data With K-Means

# Preparing the data before apply model. 

# In[40]:


#We will use this "df" DataFrame for apply model.
df = pd.DataFrame(df_train.groupby(["item_id", "shop_id", "year-month"]).item_cnt_day.sum())
df= df.reset_index().sort_values(by="year-month", ascending=True)
df.head()


# In[ ]:





# In[82]:


df= pd.merge(df, df_train, on="item_id", how="inner", )
df.head()


# In[50]:


df= df_train #We changed dataframe name for make using easier.
del_col= ["item_name", "item_category_name", "shop_name"]
for col in del_col:
    del df[col]


# Before clustering we will make standardization.

# In[51]:


# we made standardize the df["date"] and df["year-month"] columns.
df["date"]= (df["date"]-df["date"].min())/(df["date"].max()- df["date"].min()) 
# df["year-month"]= (df["year-month"]-df["year-month"].min())/(df["year-month"].max()- df["year-month"].min())


# In[52]:


scaler= StandardScaler()
X_std= scaler.fit_transform(df.drop(["year-month"], axis=1))


# We will also apply the MiniBatchK-Means clustering. 

# In[53]:


from yellowbrick.cluster import SilhouetteVisualizer


# In[59]:


mb_k_means= MiniBatchKMeans(n_clusters=10, random_state=123)
visualizer= SilhouetteVisualizer(mb_k_means, colors="yellowbrick")
visualizer.fit(X_std)
visualizer.show()


# The first thing to notice is that MiniBatchKMeans very faster than KMeans. Let's compare this models by silhouette score.

# ### 6- Applying XGBoost

# In[150]:


#We are setting the features before applying XGBoost.
X= df.drop(["item_cnt_day", "year-month"], axis=1)
y=df["item_cnt_day"]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=123)


# In[151]:


start= time.process_time()
df_xg= xgb.XGBRegressor(objective="reg:linear", seed=123)
df_xg.fit(X_train, y_train)
print(time.process_time()- start)
y_pred= df_xg.predict(X_test)


# In[153]:


ax= xgb.plot_importance(df_xg)
ax.figure.set_size_inches(20,8)
plt.show()


# In[158]:


print("MSE: {}".format(mse(y_test, y_pred)))
print("RMSE: {}".format(np.sqrt(mse(y_test, y_pred))))
print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))


# Cross Validation Of Model

# In[159]:


#Before cross validation we should transform data to DMatrix format.
df_xg= xgb.DMatrix(data=X, label=y)
params= {"objective": "reg:linear", "max_depth":5, "silent":1}
df_xg_cv= xgb.cv(dtrain=df_xg, num_boost_round=100, early_stopping_rounds=5, params=params, metrics="rmse", as_pandas=True)
display(df_xg_cv.sort_values(by="test-rmse-mean").head(5))


# Hyperparameter Tuning Of Model

# In[ ]:


#Tuning with GridSearch 
param_grid= {"penalty": ["l1", "l2"],
         "C": [10**x for x in np.arange(1,5,1)],
         #"gamma": [10**x for x in np.arange(-1,5,1)],
         #"lambda": [10**x for x in np.arange(-1,5,1)],
         #"max_depth": [1,2,3],
         "learning_rate": [0.1]
          }
df_grid= xgb.XGBRegressor(objective="reg:linear", seed=123) 
model_grid= GridSearchCV(estimator=df_grid, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
model_grid.fit(X_train,y_train)


# In[ ]:


print(model_grid.best_score_)
model_grid.best_params_


# In[ ]:


df_cv= pd.DataFrame(model_grid.cv_results_["params"])     



df_cv["mean-test-score"]= model_grid.cv_results_["mean-test-score"]
df_cv.sort_values(by="mean-test-score").head()


# ### 6- Applying RNN Model

# In[ ]:




