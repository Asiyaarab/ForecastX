import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import datetime
import plotly.express as px
import joblib
import os
import hashlib

# --- 1. SET PAGE CONFIG & CUSTOM STYLING ---
st.set_page_config(page_title="Walmart AI Forecast", layout="wide", page_icon="📈")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #0071ce; /* Walmart Blue */
    }
    .stButton>button {
        width: 100%;
        background-color: #4a90d9; /* Light Blue */
        color: #ffffff;
        font-weight: bold;
        border: none;
        height: 3em;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #3a7bc0; /* Slightly darker light blue on hover */
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CURRENCY CONVERSION ---
USD_TO_INR = 83  # adjust this rate if needed

# --- 2. SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'user_store' not in st.session_state:
    st.session_state['user_store'] = None
if 'login_role_choice' not in st.session_state:
    st.session_state['login_role_choice'] = None  # "Admin" or "User", picked via button

# --- 3. LOGIN PAGE UI ---
def login_page():
    # Load valid store/city names for validating user logins (e.g. "Mumbai" / "mumbai_manager")
    try:
        valid_cities = pd.read_csv('stores.csv')['Store'].tolist()
    except Exception:
        valid_cities = []

    # Center the login box
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)  # Add some spacing
        st.title("🔐 Secure Login")
        st.caption("Please sign in to access the Walmart Retail Intelligence Dashboard.")

        # --- STEP 1: choose role via two buttons ---
        if st.session_state['login_role_choice'] is None:
            st.write("Choose how you'd like to sign in:")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("👤 User", use_container_width=True):
                    st.session_state['login_role_choice'] = "User"
                    st.rerun()
            with b2:
                if st.button("🛡️ Admin", use_container_width=True):
                    st.session_state['login_role_choice'] = "Admin"
                    st.rerun()

        # --- STEP 2: show the login form for the chosen role ---
        else:
            role_choice = st.session_state['login_role_choice']
            st.info(f"Logging in as **{role_choice}**")

            if role_choice == "User":
                st.caption("Username: your store's city name (e.g. **Mumbai**). Password: **mumbai_manager**")

            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Sign In")

                if submit_button:
                    if role_choice == "Admin":
                        # --- HARDCODED ADMIN CREDENTIALS ---
                        if username == "admin" and password == "password":
                            st.session_state['logged_in'] = True
                            st.session_state['role'] = "admin"
                            st.session_state['user_store'] = None
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password. Please try again.")

                    elif role_choice == "User":
                        # Match username against actual city names (case-insensitive)
                        matched_city = next(
                            (city for city in valid_cities if city.lower() == username.strip().lower()),
                            None
                        )
                        expected_password = f"{username.strip().lower()}_manager"

                        if matched_city and password == expected_password:
                            st.session_state['logged_in'] = True
                            st.session_state['role'] = "user"
                            st.session_state['user_store'] = matched_city
                            st.rerun()
                        else:
                            st.error("❌ Invalid store name or password. Please try again.")

            if st.button("⬅ Back"):
                st.session_state['login_role_choice'] = None
                st.rerun()

# --- 4. MAIN APP (Only visible if logged in) ---
def main_dashboard():
    # DATA LOADING & MODEL TRAINING (cached to disk — survives app.py edits/restarts,
    # only retrains if train.csv actually changes)
    @st.cache_resource
    def load_and_train():
        train = pd.read_csv('train.csv')
        stores = pd.read_csv('stores.csv')
        features = pd.read_csv('features.csv')

        stores.rename(columns={'Store': 'City'}, inplace=True)
        stores['Store'] = range(1, len(stores) + 1)

        df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        df = df.merge(stores, on=['Store'], how='left')

        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Day'] = df['Date'].dt.day
        df['Type'] = df['Type'].map({'A': 1, 'B': 2, 'C': 3})
        df.fillna(0, inplace=True)
        df['IsHoliday'] = df['IsHoliday'].astype(int)

        features_list = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price',
                         'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                         'CPI', 'Unemployment', 'Type', 'Size', 'Year', 'Month', 'Week', 'Day']

        X = df[features_list]
        y = df['Weekly_Sales']

        # Model cache file — fingerprinted on train.csv's contents, so it only
        # retrains when the actual data changes, not when app.py code changes.
        cache_dir = ".model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        with open('train.csv', 'rb') as f:
            data_hash = hashlib.md5(f.read()).hexdigest()[:12]
        model_path = os.path.join(cache_dir, f"model_{data_hash}.joblib")
        importances_path = os.path.join(cache_dir, f"importances_{data_hash}.joblib")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # HistGradientBoostingRegressor: same idea as GradientBoostingRegressor
            # but histogram-based under the hood, so it trains dramatically faster
            # on datasets this size (400k+ rows).
            model = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42)
            model.fit(X, y)
            joblib.dump(model, model_path)

        # HistGradientBoostingRegressor has no feature_importances_ attribute,
        # so compute permutation importance once (on a small sample for speed) and cache it.
        if os.path.exists(importances_path):
            importances = joblib.load(importances_path)
        else:
            sample_idx = X.sample(n=min(3000, len(X)), random_state=42).index
            perm_result = permutation_importance(
                model, X.loc[sample_idx], y.loc[sample_idx],
                n_repeats=3, random_state=42, n_jobs=-1
            )
            importances = perm_result.importances_mean
            joblib.dump(importances, importances_path)

        dept_names = train[['Dept', 'DepartmentName']].drop_duplicates().sort_values('Dept')

        # Average Weekly_Sales split by holiday vs regular week, per Store + Dept
        sales_by_holiday = train.groupby(['Store', 'Dept', 'IsHoliday'])['Weekly_Sales'].mean().reset_index()

        # Full historical Store+Dept+Date+Sales, used for the trend line chart
        sales_history = train[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        sales_history['Date'] = pd.to_datetime(sales_history['Date'])

        return model, features, stores, features_list, dept_names, sales_by_holiday, importances, sales_history

    # Setup
    with st.spinner('🚀 Initializing AI Forecasting Engine...'):
        model, features_df, stores_df, features_list, dept_names, sales_by_holiday, importances, sales_history = load_and_train()

    # HEADER & SIDEBAR
    st.title("🏙️ Walmart Retail Intelligence")
    st.caption("Advanced Sales Forecasting using Gradient Boosting Machines")

    role = st.session_state['role']
    st.sidebar.header("🕹️ Control Panel")

    if role == "user":
        # USER ROLE: locked to their assigned store, no dropdown
        assigned_city = st.session_state['user_store']
        st.sidebar.info(f"👤 Logged in as **User**\n\nStore: **{assigned_city}**")
        selected_city = assigned_city
        store_id = stores_df[stores_df['City'] == selected_city]['Store'].values[0]
    else:
        # ADMIN ROLE: can pick any store
        st.sidebar.success("🛡️ Logged in as **Admin** — all stores visible")
        selected_city = st.sidebar.selectbox("Select Store Location", sorted(stores_df['City'].unique()))
        store_id = stores_df[stores_df['City'] == selected_city]['Store'].values[0]

    selected_dept_name = st.sidebar.selectbox("Select Department", dept_names['DepartmentName'].unique())
    dept_id = dept_names[dept_names['DepartmentName'] == selected_dept_name]['Dept'].values[0]
    prediction_date = st.sidebar.date_input("Target Date", datetime.date(2011, 11, 25))

    st.sidebar.divider()

    # SIGN OUT BUTTON
    if st.sidebar.button("🚪 Sign Out"):
        st.session_state['logged_in'] = False
        st.session_state['role'] = None
        st.session_state['user_store'] = None
        st.session_state['login_role_choice'] = None
        st.rerun()

    # PREDICTION & UI LAYOUT
    if st.sidebar.button("GENERATE FORECAST"):
        date_str = prediction_date.strftime('%Y-%m-%d')
        feat_row = features_df[(features_df['Store'] == store_id) & (features_df['Date'] == date_str)]
        store_row = stores_df[stores_df['Store'] == store_id]

        if feat_row.empty:
            st.error(f"Historical context missing for {selected_city} on this date. Please try another date between 2010 and 2012.")
        else:
            input_data = pd.DataFrame([{
                'Store': store_id, 'Dept': dept_id, 'IsHoliday': int(feat_row['IsHoliday'].values[0]),
                'Temperature': feat_row['Temperature'].values[0], 'Fuel_Price': feat_row['Fuel_Price'].values[0],
                'MarkDown1': feat_row['MarkDown1'].fillna(0).values[0], 'MarkDown2': feat_row['MarkDown2'].fillna(0).values[0],
                'MarkDown3': feat_row['MarkDown3'].fillna(0).values[0], 'MarkDown4': feat_row['MarkDown4'].fillna(0).values[0],
                'MarkDown5': feat_row['MarkDown5'].fillna(0).values[0], 'CPI': feat_row['CPI'].values[0],
                'Unemployment': feat_row['Unemployment'].values[0], 'Type': {'A': 1, 'B': 2, 'C': 3}[store_row['Type'].values[0]],
                'Size': store_row['Size'].values[0], 'Year': prediction_date.year, 'Month': prediction_date.month,
                'Week': int(prediction_date.isocalendar()[1]), 'Day': prediction_date.day
            }])

            prediction = model.predict(input_data[features_list])[0]
            prediction_inr = prediction * USD_TO_INR
            fuel_price_inr = feat_row['Fuel_Price'].values[0] * USD_TO_INR

            # Top Metric Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Sales", f"₹{prediction_inr:,.2f}")
            m2.metric("Holiday Status", "Holiday 🎉" if feat_row['IsHoliday'].values[0] else "Regular 🏠")
            m3.metric("Store Size", f"{store_row['Size'].values[0]:,} sq ft")
            m4.metric("Fuel Price", f"₹{fuel_price_inr:.2f}")

            # --- DOWNLOAD (Sidebar): this forecast result as CSV ---
            result_df = pd.DataFrame([{
                'Store': selected_city,
                'Department': selected_dept_name,
                'Target Date': date_str,
                'Predicted Sales (INR)': round(prediction_inr, 2),
                'Holiday Week': bool(feat_row['IsHoliday'].values[0]),
                'Store Size (sq ft)': int(store_row['Size'].values[0]),
                'Fuel Price (INR)': round(fuel_price_inr, 2),
                'CPI': round(feat_row['CPI'].values[0], 2),
                'Unemployment (%)': round(feat_row['Unemployment'].values[0], 2),
                'Temperature (F)': round(feat_row['Temperature'].values[0], 1),
            }])
            st.sidebar.divider()
            st.sidebar.download_button(
                label="⬇️ Download This Forecast (CSV)",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name=f"forecast_{selected_city}_{selected_dept_name}_{date_str}.csv".replace(" ", "_"),
                mime="text/csv",
                use_container_width=True
            )

            # --- ADMIN ONLY: download the full all-stores dataset (Sidebar) ---
            if role == "admin":
                full_export = sales_history.merge(
                    stores_df[['Store', 'City']], on='Store', how='left'
                ).merge(
                    dept_names, on='Dept', how='left'
                )
                full_export['Weekly_Sales_INR'] = (full_export['Weekly_Sales'] * USD_TO_INR).round(2)
                full_export = full_export[['City', 'DepartmentName', 'Date', 'Weekly_Sales_INR']].rename(
                    columns={'City': 'Store', 'DepartmentName': 'Department', 'Weekly_Sales_INR': 'Weekly Sales (INR)'}
                )
                st.sidebar.download_button(
                    label="⬇️ [Admin] Full Dataset (All Stores)",
                    data=full_export.to_csv(index=False).encode('utf-8'),
                    file_name="forecastx_all_stores_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Regular vs Holiday average sales comparison for this store + department
            hol_data = sales_by_holiday[(sales_by_holiday['Store'] == store_id) & (sales_by_holiday['Dept'] == dept_id)]
            regular_avg = hol_data[hol_data['IsHoliday'] == False]['Weekly_Sales'].mean()
            holiday_avg = hol_data[hol_data['IsHoliday'] == True]['Weekly_Sales'].mean()
            regular_avg = 0 if pd.isna(regular_avg) else regular_avg * USD_TO_INR
            holiday_avg = 0 if pd.isna(holiday_avg) else holiday_avg * USD_TO_INR

            st.divider()
            st.subheader("🏠 Regular vs 🎉 Holiday — Average Weekly Sales")
            comp_col1, comp_col2 = st.columns(2)
            comp_col1.metric("Regular Week Avg 🏠", f"₹{regular_avg:,.2f}")
            comp_col2.metric("Holiday Week Avg 🎉", f"₹{holiday_avg:,.2f}")

            comp_df = pd.DataFrame({
                'Week Type': ['Regular 🏠', 'Holiday 🎉'],
                'Average Sales (₹)': [regular_avg, holiday_avg]
            })
            comp_fig = px.bar(
                comp_df, x='Average Sales (₹)', y='Week Type', color='Week Type',
                orientation='h', text='Average Sales (₹)',
                color_discrete_map={'Regular 🏠': '#0071ce', 'Holiday 🎉': '#ffc220'},
                template="plotly_white"
            )
            comp_fig.update_traces(
                texttemplate='₹%{text:,.0f}', textposition='outside',
                width=0.5, marker_line_width=0
            )
            comp_fig.update_layout(
                showlegend=False, height=260,
                xaxis_title=None, yaxis_title=None,
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(showgrid=False, showticklabels=False),
                bargap=0.5
            )
            st.plotly_chart(comp_fig, use_container_width=True)

            # Historical weekly sales trend (line chart) for this store + department
            st.divider()
            st.subheader(f"📈 Weekly Sales Trend — {selected_city} / {selected_dept_name}")
            trend_data = sales_history[
                (sales_history['Store'] == store_id) & (sales_history['Dept'] == dept_id)
            ].sort_values('Date').copy()

            if trend_data.empty:
                st.info("No historical sales data available for this store/department combination.")
            else:
                trend_data['Weekly_Sales_INR'] = trend_data['Weekly_Sales'] * USD_TO_INR
                trend_fig = px.line(
                    trend_data, x='Date', y='Weekly_Sales_INR',
                    markers=True, template="plotly_white",
                    labels={'Weekly_Sales_INR': 'Weekly Sales (₹)', 'Date': 'Week'}
                )
                trend_fig.update_traces(line_color='#0071ce', line_width=2)

                # Compare forecast to the most recent actual week before the target date,
                # to color the star green (up) or red (down) and show a delta label.
                prior_data = trend_data[trend_data['Date'] < pd.to_datetime(date_str)]
                if not prior_data.empty:
                    last_actual = prior_data.sort_values('Date').iloc[-1]['Weekly_Sales_INR']
                    delta = prediction_inr - last_actual
                    delta_pct = (delta / last_actual * 100) if last_actual != 0 else 0
                    star_color = '#2ecc71' if delta >= 0 else '#e74c3c'
                    arrow = '▲' if delta >= 0 else '▼'
                    delta_label = f"{arrow} ₹{abs(delta):,.0f} ({delta_pct:+.1f}%) vs last week"
                else:
                    star_color = '#ffc220'
                    delta_label = "No prior week to compare"

                # Mark the predicted forecast point on the same trend line
                trend_fig.add_scatter(
                    x=[pd.to_datetime(date_str)], y=[prediction_inr],
                    mode='markers+text', marker=dict(color=star_color, size=16, symbol='star',
                                                       line=dict(color='white', width=1)),
                    text=[delta_label], textposition='top center',
                    name='Forecast'
                )
                trend_fig.update_layout(height=420, showlegend=True)
                if not prior_data.empty:
                    trend_caption = (
                        f"🟢 Green star = forecast is **up** vs last actual week &nbsp;|&nbsp; "
                        f"🔴 Red star = forecast is **down** vs last actual week"
                    )
                    st.caption(trend_caption)
                st.plotly_chart(trend_fig, use_container_width=True)

            # Dashboard Body
            st.divider()
            c1, c2 = st.columns([2, 1])

            with c1:
                st.subheader("🤖 Model Decision Factors")
                feat_imp = pd.DataFrame({'Feature': features_list, 'Importance': importances}).sort_values(by='Importance', ascending=True).tail(8)
                fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                             color='Importance', color_continuous_scale='Blues', template="plotly_white")
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader(f"📊 {selected_city} Context")
                st.info(f"**CPI:** {feat_row['CPI'].values[0]:.2f}")
                st.info(f"**Unemployment:** {feat_row['Unemployment'].values[0]:.2f}%")
                st.info(f"**Temp:** {feat_row['Temperature'].values[0]:.1f}°F")

    else:
        # Welcome Screen
        try:
            st.image("sales_forecasting_banner.jpg", use_container_width=True)
        except:
            st.image("https://images.unsplash.com/photo-1534452285072-c0c1c5775222?auto=format&fit=crop&q=80&w=2070", use_container_width=True)
        st.warning("👈 Please set your parameters in the sidebar and click Generate Forecast.")

# --- 5. APP ROUTING LOGIC ---
if not st.session_state['logged_in']:
    login_page()
else:
    main_dashboard()