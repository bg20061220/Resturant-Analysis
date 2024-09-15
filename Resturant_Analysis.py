import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset as data and another copy with no changes as original_data
original_data = pd.read_csv('zomato_sales.csv')
data = pd.read_csv('zomato_sales.csv')

# Removing rows with missing values in 'online_order' column
data = data.dropna(subset=['online_order'])

# Replacing 'Yes' and 'No' values in 'online_order' with 1 and 0 respectively
data['online_order'] = data['online_order'].replace({'Yes': 1, 'No': 0})

# Removing rows with missing values in 'book_table' column and replace 'Yes'/'No' with 1/0
data = data.dropna(subset=['book_table'])
data['book_table'] = data['book_table'].replace({'Yes': 1, 'No': 0})

# Cleaning up the 'rate' column by converting it to float after splitting from '/5' format
data = data.dropna(subset=['rate'])
data['rate'] = data['rate'].astype(str)
data = data[data['rate'].str.match(r'^\d')]
data['rate'] = data['rate'].str.split('/').str[0].astype('float')

# Converting 'approx_cost(for two people)' to numeric after cleaning commas
data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(str)
data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',', '')
data['approx_cost(for two people)'] = pd.to_numeric(data['approx_cost(for two people)'], errors='coerce')

# Droping rows with missing values in 'approx_cost(for two people)' and 'sales(in CAD) in 2023'
data = data.dropna(subset=['approx_cost(for two people)', 'sales(in CAD) in 2023'])

# Adding a new column for sales data (randomly generated for demonstration purposes)
np.random.seed(42)
data['sales(in CAD) in 2023'] = np.random.randint(90000, 270000, len(data))

# Prepared data for correlation analysis between money (approx_cost) and rating
correlation = data['approx_cost(for two people)'].corr(data['rate'])
print(f"The correlation between money and rating is {correlation}")

# Finding the top 25 restaurants based on cost and rating
expensive_restaurants = data.sort_values(by=['approx_cost(for two people)', 'rate'], ascending=[False, False])
top25_expensive_and_rating = expensive_restaurants.head(25)
city_counts = top25_expensive_and_rating['listed_in(city)'].value_counts()

#  The best city from the top 25 restaurants
best_city = city_counts.idxmax()
best_city_count = city_counts.max()
print(f"Best city is {best_city} with {best_city_count} out of 25 restaurants")

# Finding mean rating and price for Malleshwaram and Banashankari
malleshwaram = data[data['listed_in(city)'] == 'Malleshwaram']
malleshwaram_mean_rating = malleshwaram['rate'].mean()
malleshwaram_mean_price = malleshwaram['approx_cost(for two people)'].mean()

banashankari = data[data['listed_in(city)'] == 'Banashankari']
banashankari_mean_rating = banashankari['rate'].mean()
banashankari_mean_price = banashankari['approx_cost(for two people)'].mean()

print(f"Mean cost in Banashankari is {banashankari_mean_price}, and in Malleshwaram it is {malleshwaram_mean_price}")

# The city with the most online ordering options
online_options = data[data['online_order'] == 1]
city_counts = online_options['listed_in(city)'].value_counts()
best_online_city = city_counts.idxmax()
best_online_city_count = city_counts.max()
print(f"The city with the most online options is {best_online_city} with {best_online_city_count} options")

# Comparing ratings between restaurants that offer and do not offer delivery
rest_without_delivery = data[data['online_order'] == 0]
rest_without_delivery_mean = rest_without_delivery['rate'].mean()

rest_with_delivery = data[data['online_order'] == 1]
rest_with_delivery_mean = rest_with_delivery['rate'].mean()

print(f"Restaurants offering delivery have a mean rating of {rest_with_delivery_mean}, while those without delivery have a rating of {rest_without_delivery_mean}")

# The city with the most votes and highest ratings
rest_by_votes_and_ratings = data.sort_values(by=['votes', 'rate'], ascending=[False, False])
top100 = rest_by_votes_and_ratings.head(100)
top_city_count = top100['listed_in(city)'].value_counts()
top_city = top_city_count.idxmax()
top_city_occurrence = top_city_count.max()

print(f"City with the most votes and best rating is {top_city} with {top_city_occurrence} occurrences in the top 100")

# Performing Linear Regression on 'approx_cost(for two people)' and 'sales(in CAD) in 2023'
X = data[['approx_cost(for two people)']]  # Independent Variable
Y = data[['sales(in CAD) in 2023']]  # Dependent Variable

# Spliting dataset into training set (80%) and testing set (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initializing the Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, Y_train)

# Predicting the sales using test data
Y_pred = model.predict(X_test)

#  Mean Squared Error and R-squared
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Plotting the data
plt.scatter(X_test, Y_test, color=(0, 0, 1), label='Actual data points for test values')
plt.plot(X_test, Y_pred, color=(1, 0, 0), label="Predicted Data on test Values")
plt.title("Approx Cost Vs Sales Analysis")
plt.xlabel("Approx Cost")
plt.ylabel("Sales")
plt.legend()
plt.show()

#  R2 and MSE values
print(f"R-squared: {r2}")  # R2 = -9.576842916558803e-05
print(f"MSE: {mse}")  # mse = 2700360008.670263

# Creating a pie chart to show the sales share by restaurant types
# Grouping the data by 'rest_type' and summing up the sales for each type
sales_by_rest_type = data.groupby('rest_type')['sales(in CAD) in 2023'].sum()

# Sorting the sales values in descending order
sales_by_rest_type = sales_by_rest_type.sort_values(ascending=False)

# Selecting the top 10 restaurant types by sales
top_10_sales = sales_by_rest_type[:10]

# Summing the sales for all other restaurant types outside of the top 10
other_sales = sales_by_rest_type[10:].sum()

# Adding the "others" category to account for the remaining sales
top_10_sales['others'] = other_sales

# Defining colors using Seaborn's bright color palette
colors = sns.color_palette('bright', len(top_10_sales) + 1)

# Creating the pie chart
plt.figure(figsize=(40, 30))
plt.pie(top_10_sales, labels=top_10_sales.index, colors=colors, autopct='%1.1f%%', startangle=140)

# Adding a title to the pie chart
plt.title("Sales by Restaurant Type")

# Ensuring the pie chart is displayed as a circle (equal aspect ratio)
plt.axis('equal')

# Displaying the pie chart
plt.show()
