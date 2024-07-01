from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
df = pd.read_csv("hotel_bookings.csv")
print(df.head())

'''
CLEANING
'''

# CONCLUSION
# agent/company columns have too much null values  so that we delete that columns

df.drop(["agent", "company"], axis=1, inplace=True)
print(df.shape)

# DROP_NULL VALUES >--already  we have alot of rows and if we have a little null_values
# then we fill but we have lots of null_values we cant take right fill_values soo thats why -->drop.
print(df.isna().sum())
df = df.dropna()
df = df.reset_index(drop=True)


'''
ANALYZE EACH COLUMN WITH RESPECT TO DEPENDENT COLUMN
'''


'''
RESERVATION_STATUS_DATE
'''
print(df['reservation_status_date'].dtypes)
split_values = df['reservation_status_date'].str.split("-", expand=True)
df['reservation_year'] = split_values[0].astype("int32")
df['reservation_month'] = split_values[1].astype("int32")
df['reservation_date'] = split_values[2].astype("int32")

# drop the old column ,bcz i extract the values so no need that column.
df.drop(['reservation_status_date'], axis=1, inplace=True)


# make new column resevation_year
print(df['reservation_year'].dtypes)
print(df['reservation_year'].value_counts())
plt.hist(df['reservation_year'], color='r')
plt.title("distribution_reservation_year")
plt.xlabel("year")
plt.ylabel("frequency")
plt.show()

# reservation_month
print(df['reservation_month'].dtypes)
print(df['reservation_month'].value_counts())
plt.hist(df['reservation_month'], color='b')
plt.title("distribution_reservation_month")
plt.xlabel("month")
plt.ylabel("frequency")
plt.show()

# df['new']=df['reservation_month'].replace({7:1,8:1,10:1,4:2,3:2,1:2,5:2,2:3,6:3,9:3,11:4,12:4})
# print(df['new'].value_counts())
# plt.hist(df['new'],color='b')
# plt.show()

# resevation_date
print(df['reservation_date'].dtypes)
print(df['reservation_date'].value_counts())
plt.hist(df['reservation_date'], color='g')
plt.title("distribution_reservation_date")
plt.xlabel("date")
plt.ylabel("frequency")
plt.show()


'''
RESERVATION_STATUS
'''
print(df['reservation_status'].dtypes)
print(df['reservation_status'].value_counts())
plt.hist(df['reservation_status'], color='purple')
plt.title("reservation_status")
plt.xlabel("status_type")
plt.ylabel("distribution")
plt.show()
l=[]
for i in range(len(df)):
    try: 
        int(df.iloc[i,-4])
    except:
        l.append(i)

'''
TOTAL_SPECIALS_REQUESTS
'''

print(df['total_of_special_requests'].dtypes)
print(df['total_of_special_requests'].value_counts())
seaborn.violinplot(df['total_of_special_requests'], color='b')
plt.title("total_specials_requests_rate")
plt.show()

# conclusion==> 4 and 5 catgories are same so  i merged that catgories.

'''
required_car_parking_spaces
'''
print(df['required_car_parking_spaces'].dtypes)
print(df['required_car_parking_spaces'].value_counts())
plt.hist(df['required_car_parking_spaces'], color='purple')
plt.title("required_car_parking_spaces")
plt.xlabel("values")
plt.ylabel("distribution")
plt.show()

'''
adr
'''
#  it is not impact on depenedent so we drop it
df.drop(columns=['adr'], inplace=True)


'''
customer_type
'''
print(df['customer_type'].dtypes)
print(df['customer_type'].value_counts())
plt.hist(df['customer_type'], color='orange')
plt.title("cutomers_type")
plt.xlabel("categories")
plt.ylabel("distribution")
plt.show()


'''
days_in_waiting_list
'''
print(df['days_in_waiting_list'].dtypes)
plt.hist(df['days_in_waiting_list'], color='blue')
plt.title("days_in_waiting_list")
plt.xlabel("values")
plt.ylabel("distribution")
plt.show()


'''
deposit_type
'''
print(df['deposit_type'].dtypes)
print(df['deposit_type'].value_counts())
plt.hist(df['deposit_type'], color='purple')
plt.title("deposite_type")
plt.xlabel("deposite_type")
plt.ylabel("distribution")
plt.show()


'''
booking_changes
'''
print(df['booking_changes'].dtypes)
seaborn.violinplot(df['booking_changes'], color='red')
plt.title("booking_changes")
plt.xlabel("booking_change")
plt.ylabel("distribution")
plt.show()

'''
assigned_room_type
'''
print(df['assigned_room_type'].dtypes)
print(df['assigned_room_type'].value_counts())

for x in df['assigned_room_type'].unique():
    f = df['assigned_room_type'] == x
    plt.violinplot(df.loc[f, 'is_canceled'])
    plt.title(x)
    plt.ticklabel_format(style="plain")
    plt.show()
f = df["assigned_room_type"] == "P"
print(df[f].index)
df.drop(index=[72475, 72476], axis=0, inplace=True)
df.reset_index(inplace=True)
# catgories=L and P is no any impacct its total null soo i drop it.
f1 = df["assigned_room_type"] == "L"
print(df[f1].index)
df.drop(index=[14086], axis=0, inplace=True)
df.reset_index(inplace=True)

# we have a lots of catgoris in assigned room type and its some similar so i merge it that catgories
df['assigned_room_type'] = df['assigned_room_type'].replace(
    {'A': 'A', 'B': 'B', 'F': 'B', 'G': 'B', 'E': 'B', 'D': 'B', 'C': 'B', 'K': 'C', 'I': 'C', 'H': 'D'})
plt.hist(df['assigned_room_type'], color='purple')
plt.title("assigned_room_type")
plt.xlabel("room_type")
plt.ylabel("distribution")
plt.show()


'''
reserved_room_type
'''
print(df['reserved_room_type'].dtypes)
print(df['reserved_room_type'].value_counts())
for x in df['reserved_room_type'].unique():
    f = df['reserved_room_type'] == x
    plt.violinplot(df.loc[f, 'is_canceled'])
    plt.title(x)
    plt.ticklabel_format(style="plain")
    plt.show()

f3 = df["reserved_room_type"] == "L"
print(df[f3].index)
df.drop(index=[353, 503, 910, 14175, 15404], axis=0, inplace=True)
df.reset_index(inplace=True)

# we have a lots of catgoris in assigned room type and its some similar so i merge it that catgories
df['reserved_room_type'] = df['reserved_room_type'].replace(
    {'A': 'A', 'B': 'C', 'F': 'C', 'G': 'C', 'E': 'C', 'D': 'C', 'C': 'C', 'H': 'B', 'D': 'B', 'l': 'B', 'L': 'D'})
plt.hist(df['reserved_room_type'], color='purple')
plt.title("reserved_room_type")
plt.xlabel("room_type")
plt.ylabel("distribution")
plt.show()


'''
previous_bookings_not_canceled
'''
print(df['previous_bookings_not_canceled'].dtypes)
print(df['previous_bookings_not_canceled'].value_counts())
seaborn.violinplot(df['previous_bookings_not_canceled'], color='red')
plt.title("previous_bookings_not_canceled")
plt.xlabel("booking")
plt.ylabel("distribution")
plt.show()


'''
previous_cancellations
'''
print(df['previous_cancellations'].dtypes)
print(df['previous_cancellations'].value_counts())
seaborn.violinplot(df['previous_cancellations'], color='red')
plt.title("previous_cancellations")
plt.show()

'''
is_repeated_guest
'''
print(df['is_repeated_guest'].dtypes)
print(df['is_repeated_guest'].value_counts())
plt.hist(df['is_repeated_guest'])
plt.title("is_repeated_guest")
plt.xlabel("guest_stay_repeated")
plt.ylabel("distribution")
plt.show()


'''
distribution_channel
'''
print(df['distribution_channel'].dtypes)
print(df['distribution_channel'].value_counts())

f = df['distribution_channel'] == 'Undefined'
print(df[f].index)
df.drop(index=[14185], inplace=True)
df.reset_index(inplace=True)
# undifined  catgorie only(1) onevalue which is outlier so delete

plt.hist(df['distribution_channel'], color='orange')
plt.title("distribution_channel")
plt.xlabel("channel")
plt.ylabel("distribution")
plt.show()


'''
market_segment
'''

print(df['market_segment'].dtypes)
print(df['market_segment'].value_counts())

for x in df['market_segment'].unique():
    f = df['market_segment'] == x
    plt.violinplot(df.loc[f, 'is_canceled'])
    plt.title(x)
    plt.ticklabel_format(style="plain")
    plt.show()

df['market_segment'] = df['market_segment'].replace(
    {'Direct': 'Direct', 'Corporate': 'Direct', 'Complementary': 'Direct', 'Offline TA/TO': 'Oflline TA'})
plt.hist(df['market_segment'], color='purple', width=0.6, align='mid')
plt.title("market_segment")
plt.xlabel("market_type")
plt.ylabel("distribution")
plt.show()


'''

country

'''
# print(df['country'].dtypes)
# print(df['country'].value_counts())
# for x in df['country'].unique():
#     f=df['country']==x
#     plt.violinplot(df.loc[f,'is_canceled'])
#     plt.title(x)
#     plt.ticklabel_format(style="plain")
#     plt.show()
# plt.hist(df['country'],color='pink')
# plt.title("country")
# plt.show()
# i drop country catgories  bcz lots of catgories(177) which create noise.
df.drop(columns=['country'], inplace=True)


'''
meal
'''
print(df['meal'].dtypes)
print(df['meal'].value_counts())
plt.hist(df['meal'], color='pink')
plt.title("meal")
plt.xlabel("meal_catgories")
plt.ylabel("frequency")
plt.show()

'''
babies
'''
print(df['babies'].dtypes)
print(df['babies'].value_counts())
plt.hist(df['babies'], color='orange')
plt.xlabel("values")
plt.ylabel("frequency")
plt.show()


'''
children
'''
print(df['children'].dtypes)
print(df['children'].value_counts())
plt.hist(df['children'], color='blue')
plt.title("children")
plt.xlabel("values")
plt.ylabel("frequency")
plt.show()

'''
adults
'''
print(df['adults'].dtypes)
print(df['adults'].value_counts())

plt.hist(df['adults'], color='yellow')
plt.title("adults")
plt.xlabel("values")
plt.ylabel("distribution")
plt.show()


'''
stays_in_week_nights
'''
print(df['stays_in_week_nights'].dtypes)
print(df['stays_in_week_nights'].value_counts)
plt.hist(df['stays_in_week_nights'], color='green')
plt.title("stays_in_week_nights")
plt.xlabel("values")
plt.ylabel("distribution")
plt.show()


'''
stays_in_weekend_nights
'''
print(df['stays_in_weekend_nights'].dtypes)
print(df['stays_in_weekend_nights'].value_counts)
plt.hist(df['stays_in_weekend_nights'], color='pink')
plt.title("stays_in_weekend_nights")
plt.xlabel("values")
plt.ylabel("distribution")
plt.show()


'''
arrival_date_day_of_month
'''
print(df['arrival_date_day_of_month'].dtypes)
print(df['arrival_date_day_of_month'].value_counts)
plt.hist(df['arrival_date_day_of_month'], color='orange')
plt.title("arrival_date_day_of_month")
plt.xlabel("months")
plt.ylabel("distribution")
plt.show()

'''
arrival_date_week_number
'''
print(df['arrival_date_week_number'].dtypes)
print(df['arrival_date_week_number'].value_counts)
plt.hist(df['arrival_date_week_number'], color='yellow')
plt.title("arrival_date_week_number")
plt.xlabel("week_numbers")
plt.ylabel("distribution")
plt.show()

'''
arrival_date_month
'''
print(df['arrival_date_month'].dtypes)
print(df['arrival_date_month'].unique())
print(df['arrival_date_month'].value_counts)
plt.hist(df['arrival_date_month'], color='pink')
plt.title("arrival_date_month")
plt.xlabel("months")
plt.ylabel("distribution")
plt.show()


'''
arrival_date_year
'''
print(df['arrival_date_year'].dtypes)
print(df['arrival_date_year'].value_counts())
plt.hist(df['arrival_date_year'], color='blue')
plt.title("arrival_date_year")
plt.xlabel("years")
plt.ylabel("distribution")
plt.show()


'''
lead_time
'''
print(df['lead_time'].dtypes)
print(df['lead_time'].value_counts())
plt.hist(df['lead_time'], color='green')
plt.title("lead_time")
plt.xlabel("time")
plt.ylabel("distribution")
plt.show()


'''
hotel
'''
print(df['hotel'].dtypes)
print(df['hotel'].value_counts())
plt.hist(df['hotel'], color='blue')
plt.title("hotel")
plt.xlabel("hotels_type")
plt.ylabel("distribution")
plt.show()


df.drop(columns=['index'], inplace=True)
df.drop(columns=['level_0'], inplace=True)

'''
is_canceled

'''
print(df['is_canceled'].dtypes)
print(df['is_canceled'].value_counts())
plt.hist(df['is_canceled'], color='orange')
plt.title("is_calcelation_rate")
plt.xlabel("cancelation_distriution")
plt.ylabel("distribution")
plt.show()


'''
STATICAL_ANALYSIS
'''
df.info()


category = df[['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
               'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status',
               'arrival_date_month']]

numeric = df[['lead_time', 'is_canceled', 'arrival_date_year', 'arrival_date_day_of_month',
              'arrival_date_week_number', 'stays_in_week_nights', 'stays_in_weekend_nights',
              'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
              'previous_bookings_not_canceled', 'days_in_waiting_list', 'reservation_month', 'reservation_date',
              "booking_changes","total_of_special_requests", "required_car_parking_spaces"]]


from sklearn.preprocessing import LabelEncoder

encoder1=LabelEncoder()
category['hotel']=encoder1.fit_transform(df['hotel'])
encoder2=LabelEncoder()
category['meal']=encoder2.fit_transform(category['meal'])
encoder3=LabelEncoder()
category['market_segment']=encoder3.fit_transform(category['market_segment'])
encoder4=LabelEncoder()
category['distribution_channel']=encoder4.fit_transform(category['distribution_channel'])
encoder5=LabelEncoder()
category['reserved_room_type']=encoder5.fit_transform(category['reserved_room_type'])
encoder6=LabelEncoder()
category['assigned_room_type']=encoder6.fit_transform(category['assigned_room_type'])
encoder7=LabelEncoder()
category['deposit_type']=encoder7.fit_transform(category['deposit_type'])
encoder8=LabelEncoder()
category['customer_type']=encoder8.fit_transform(category['customer_type'])
encoder9=LabelEncoder()
category['reservation_status']=encoder9.fit_transform(category['reservation_status'])
encoder10=LabelEncoder()
category['arrival_date_month']=encoder10.fit_transform(category['arrival_date_month'])

category.head()
numeric.head()









#  we use 'pearson 'correlation on is numeric  and another is also numeric
'''
FEATURE SELECTION
'''

corr = numeric.corr()
corr

seaborn.heatmap(corr)
#  these all are a very bad correaltion with respect dependent.
df.drop(columns=['reservation_year', "booking_changes",
        "total_of_special_requests", "required_car_parking_spaces"], inplace=True)


# ANOVA use in one is catgorical and another is numeric

# Select K best is used to sort the best columns according to test
# value passed as parameter
sk = SelectKBest(f_classif, k=10)
# k value represent the number of columns we want
# This value depends upon our domain knowledge

result = sk.fit_transform(category, numeric['is_canceled'])
# In result ,there are top n columns with highest f score
print(result)
print(sk.scores_)


'''
OUTLIERS
'''

plt.scatter(numeric['stays_in_week_nights'], numeric['is_canceled'])
plt.show()

for x in numeric.columns:
    plt.hist(numeric[x])
    plt.title(x)
    plt.show()


def z_score(column):
    mean = column.mean()
    std = column.std()
    z = np.abs((column-mean)/std)
    return column[z > 3]

outliers1 = z_score(numeric['lead_time'])
outliers2 = z_score(numeric['stays_in_week_nights'])
outliers3 = z_score(numeric['adults'])
outliers4 = z_score(numeric['children'])
outliers5 = z_score(numeric['babies'])
outliers6 = z_score(numeric['is_repeated_guest'])
outliers7 = z_score(numeric['previous_cancellations'])
outliers8 = z_score(numeric['previous_bookings_not_canceled'])
outliers9 = z_score(numeric['arrival_date_week_number'])
outliers10 = z_score(numeric['stays_in_weekend_nights'])
outliers11 = z_score(numeric['days_in_waiting_list'])

# drop the row of outliers

numeric = numeric.drop(index=outliers1.index).reset_index(drop=True)
category = category.drop(index=outliers1.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers2.index).reset_index(drop=True)
category = category.drop(index=outliers2.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers3.index).reset_index(drop=True)
category = category.drop(index=outliers3.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers4.index).reset_index(drop=True)
category = category.drop(index=outliers4.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers5.index).reset_index(drop=True)
category = category.drop(index=outliers5.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers6.index).reset_index(drop=True)
category = category.drop(index=outliers6.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers7.index).reset_index(drop=True)
category = category.drop(index=outliers7.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers8.index).reset_index(drop=True)
category = category.drop(index=outliers8.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers9.index).reset_index(drop=True)
category = category.drop(index=outliers9.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers10.index).reset_index(drop=True)
category = category.drop(index=outliers10.index).reset_index(drop=True)

numeric = numeric.drop(index=outliers11.index).reset_index(drop=True)
category = category.drop(index=outliers11.index).reset_index(drop=True)

print(len(numeric), len(category))




