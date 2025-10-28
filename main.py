import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Step 1: Load Dataset
df = pd.read_csv('sales_data.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Step 2: Compute Total Revenue
df['Total_Revenue'] = df['Units_Sold'] * df['Unit_Price']

# ------------------------------------------
# Step 3: Interactive Visualizations
# ------------------------------------------

# 1️⃣ Daily Revenue Trend
daily_revenue = df.groupby('Date')['Total_Revenue'].sum().reset_index()
fig1 = px.line(daily_revenue, x='Date', y='Total_Revenue',
               title='📈 Daily Revenue Trend',
               markers=True)
fig1.show()

# 2️⃣ Revenue by Category
category_sales = df.groupby('Category')['Total_Revenue'].sum().reset_index()
fig2 = px.bar(category_sales, x='Category', y='Total_Revenue',
              color='Category', title='💰 Revenue by Category')
fig2.show()

# 3️⃣ Units Sold vs Total Revenue per Product
fig3 = px.scatter(df, x='Units_Sold', y='Total_Revenue', color='Product',
                  size='Total_Revenue', hover_data=['Date'],
                  title='📊 Units Sold vs Revenue per Product')
fig3.show()

# ------------------------------------------
# Step 4: Time Series Forecasting
# ------------------------------------------

# Prepare daily time series
daily_revenue.set_index('Date', inplace=True)

# Fit Simple Exponential Smoothing (SES) Model
model = SimpleExpSmoothing(daily_revenue['Total_Revenue']).fit(smoothing_level=0.2, optimized=False)

# Forecast next 7 days
forecast = model.forecast(7)
print("Next 7 Days Forecast:\n", forecast)

# ------------------------------------------
# Step 5: Plot Forecast
# ------------------------------------------

fig4 = go.Figure()

# Actual Revenue
fig4.add_trace(go.Scatter(
    x=daily_revenue.index,
    y=daily_revenue['Total_Revenue'],
    mode='lines+markers',
    name='Actual Revenue'
))

# Forecasted Revenue
future_dates = pd.date_range(start=daily_revenue.index[-1] + pd.Timedelta(days=1), periods=7)
fig4.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    mode='lines+markers',
    name='Forecasted Revenue'
))

fig4.update_layout(
    title='📆 Daily Revenue Forecast (Next 7 Days)',
    xaxis_title='Date',
    yaxis_title='Revenue'
)