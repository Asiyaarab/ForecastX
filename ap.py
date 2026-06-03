import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import datetime
import plotly.express as px

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
        background-color: #ffc220; /* Walmart Yellow */
        color: #0071ce;
        font-weight: bold;
        border: none;
        height: 3em;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #0071ce;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- 3. LOGIN PAGE UI ---
def login_page():
    # Center the login box
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True) # Add some spacing
        st.title("🔐 Secure Login")
        st.caption("Please sign in to access the Walmart Retail Intelligence Dashboard.")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            # type="password" hides the text
            password = st.text_input("Password", type="password") 
            submit_button = st.form_submit_button("Sign In")
            
            if submit_button:
                # --- HARDCODED CREDENTIALS ---
                if username == "admin" and password == "password":
                    st.session_state['logged_in'] = True
                    st.rerun() # Refresh the app to load the dashboard
                else:
                    st.error("❌ Invalid username or password. Please try again.")

# --- 4. MAIN APP (Only visible if logged in) ---
def main_dashboard():
    # DATA LOADING & MODEL TRAINING (Cached)
    @st.cache_data
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
        
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        return model, features, stores, features_list

    # Setup
    with st.spinner('🚀 Initializing AI Forecasting Engine...'):
        model, features_df, stores_df, features_list = load_and_train()

    # HEADER & SIDEBAR
    st.title("🏙️ Walmart Retail Intelligence")
    st.caption("Advanced Sales Forecasting using Gradient Boosting Machines")

    st.sidebar.header("🕹️ Control Panel")
    
    selected_city = st.sidebar.selectbox("Select Store Location", sorted(stores_df['City'].unique()))
    store_id = stores_df[stores_df['City'] == selected_city]['Store'].values[0]
    dept_id = st.sidebar.number_input("Department Number", 1, 99, 1)
    prediction_date = st.sidebar.date_input("Target Date", datetime.date(2011, 11, 25))

    st.sidebar.divider()
    
    # SIGN OUT BUTTON
    if st.sidebar.button("🚪 Sign Out"):
        st.session_state['logged_in'] = False
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

            # Top Metric Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Sales", f"${prediction:,.2f}")
            m2.metric("Holiday Status", "Holiday 🎄" if feat_row['IsHoliday'].values[0] else "Regular 🏠")
            m3.metric("Store Size", f"{store_row['Size'].values[0]:,} sq ft")
            m4.metric("Fuel Price", f"${feat_row['Fuel_Price'].values[0]:.2f}")

            # Dashboard Body
            st.divider()
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader("🤖 Model Decision Factors")
                importances = model.feature_importances_
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