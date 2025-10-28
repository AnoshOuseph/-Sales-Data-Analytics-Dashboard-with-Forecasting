import pandas as pd

# Load CSV
df = pd.read_csv('sales_data.csv')

# Preview data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
df['Total_Revenue'] = df['Units_Sold'] * df['Unit_Price']
print(df)
category_sales = df.groupby('Category')['Total_Revenue'].sum()
print(category_sales)
product_units = df.groupby('Product')['Units_Sold'].sum()
print(product_units)
df['Date'] = pd.to_datetime(df['Date'])
daily_revenue = df.groupby('Date')['Total_Revenue'].sum()
print(daily_revenue)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Bar chart: Sales by category
category_sales.plot(kind='bar', title='Total Revenue by Category')
plt.ylabel('Revenue')
plt.show()

# 2. Line chart: Daily revenue trend
daily_revenue.plot(kind='line', title='Daily Revenue Trend')
plt.ylabel('Revenue')
plt.show()

# 3. Seaborn: Units sold per product
sns.barplot(x=product_units.index, y=product_units.values)
plt.title('Total Units Sold per Product')
plt.show()

df.to_csv('sales_data_cleaned.csv', index=False)