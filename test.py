# Quick eval script — run this once locally, paste numbers into README
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import math

train = pd.read_csv('train.csv')
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')

# stores.csv uses city names in the Store column — match app.py workaround
stores.rename(columns={'Store': 'City'}, inplace=True)
stores['Store'] = range(1, len(stores) + 1)

df = train.merge(features, on=['Store','Date','IsHoliday'], how='left')
df = df.merge(stores, on='Store', how='left')
df['Date'] = pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year; df['Month']=df['Date'].dt.month
df['Week']=df['Date'].dt.isocalendar().week.astype(int)
df['Day']=df['Date'].dt.day
df['Type']=df['Type'].map({'A':1,'B':2,'C':3})
df.fillna(0, inplace=True)
df['IsHoliday']=df['IsHoliday'].astype(int)

feat = ['Store','Dept','IsHoliday','Temperature','Fuel_Price','MarkDown1','MarkDown2',
        'MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment','Type','Size',
        'Year','Month','Week','Day']
X = df[feat]; y = df['Weekly_Sales']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

m = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
p = m.predict(X_te)

mae = mean_absolute_error(y_te, p)
rmse = math.sqrt(((y_te - p) ** 2).mean())
smape = 100 * np.mean(2 * np.abs(p - y_te) / (np.abs(p) + np.abs(y_te) + 1e-9))

# naive baseline = last week's sales
naive_pred = X_te.copy()  # placeholder
# simpler baseline check:
naive_mae = mean_absolute_error(y_te, [y_tr.mean()] * len(y_te))

print(f"MAE: {mae:.2f}")
print(f"SMAPE: {smape:.2f}%")
print(f"RMSE: {rmse:.2f}")

# Top 5 features
imp = pd.DataFrame({'Feature': feat, 'Importance': m.feature_importances_}).sort_values('Importance', ascending=False)
print(imp.head(5).to_string(index=False))