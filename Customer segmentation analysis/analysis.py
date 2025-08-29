# customer_segmentation_2026.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------------------------------
# 1. Streamlit Config
# ----------------------------------------
st.set_page_config(page_title="AI-Powered Customer Segmentation Dashboard", layout="wide")
st.title("🤖 2026 AI-Powered Customer Segmentation Dashboard")
st.markdown("Use data, ML, and predictive insights to optimize marketing & retention strategies.")

# ----------------------------------------
# 2. Create / Load Sample Dataset
# ----------------------------------------
np.random.seed(42)
n_customers = 500
data = pd.DataFrame({
    "customer_id": range(1, n_customers + 1),
    "total_purchase_amount": np.random.randint(50, 5000, n_customers),
    "number_of_orders": np.random.randint(1, 40, n_customers),
    "last_purchase_days_ago": np.random.randint(1, 730, n_customers),
})

# Calculate average order value
data["avg_order_value"] = data["total_purchase_amount"] / data["number_of_orders"]

# Simulate churn probability (for predictions)
data["churn_probability"] = np.random.uniform(0, 1, n_customers)

# ----------------------------------------
# 3. Feature Scaling + KMeans Segmentation
# ----------------------------------------
features = data[['total_purchase_amount', 'number_of_orders', 'avg_order_value']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
data['segment'] = kmeans.fit_predict(scaled_features)

# Map segments dynamically based on average purchase
segment_order = data.groupby('segment')['total_purchase_amount'].mean().sort_values().index
segment_labels = {segment_order[0]: "Low Value", segment_order[1]: "Medium Value", segment_order[2]: "High Value"}
data['segment'] = data['segment'].map(segment_labels)

# ----------------------------------------
# 4. Predictive Modeling: Next Purchase Likelihood
# ----------------------------------------
data['recent_purchase'] = np.where(data['last_purchase_days_ago'] < 90, 1, 0)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, data['recent_purchase'])
data['predicted_next_purchase'] = model.predict_proba(features)[:, 1]

# ----------------------------------------
# 5. Sidebar Filters
# ----------------------------------------
segment_filter = st.sidebar.multiselect(
    "Select Customer Segment(s):",
    options=data['segment'].unique(),
    default=data['segment'].unique()
)
filtered_data = data[data['segment'].isin(segment_filter)]

# ----------------------------------------
# 6. KPI Metrics
# ----------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🧑‍🤝‍🧑 Total Customers", len(filtered_data))
col2.metric("💰 Total Revenue", f"${filtered_data['total_purchase_amount'].sum():,.0f}")
col3.metric("📦 Total Orders", filtered_data['number_of_orders'].sum())
col4.metric("🔮 Avg Next Purchase Probability", f"{filtered_data['predicted_next_purchase'].mean()*100:.1f}%")

# ----------------------------------------
# 7. Visualizations
# ----------------------------------------
# 7.1 Customer Segment Distribution
fig_pie = px.pie(
    filtered_data,
    names='segment',
    title='Customer Segment Distribution',
    color='segment',
    color_discrete_map={"High Value": "gold", "Medium Value": "royalblue", "Low Value": "lightgray"}
)
st.plotly_chart(fig_pie, use_container_width=True)

# 7.2 Revenue per Segment
fig_bar = px.bar(
    filtered_data.groupby('segment')['total_purchase_amount'].sum().reset_index(),
    x='segment',
    y='total_purchase_amount',
    color='segment',
    title='Total Revenue per Segment'
)
st.plotly_chart(fig_bar, use_container_width=True)

# 7.3 Orders vs Revenue Scatter
fig_scatter = px.scatter(
    filtered_data,
    x='number_of_orders',
    y='total_purchase_amount',
    color='segment',
    size='avg_order_value',
    hover_data=['customer_id', 'predicted_next_purchase'],
    title='Orders vs Revenue (Segmented)'
)
st.plotly_chart(fig_scatter, use_container_width=True)

# 7.4 Predicted Next Purchase Likelihood
fig_line = go.Figure()
for segment in filtered_data['segment'].unique():
    seg_data = filtered_data[filtered_data['segment'] == segment]
    fig_line.add_trace(go.Histogram(
        x=seg_data['predicted_next_purchase'],
        name=segment,
        opacity=0.7
    ))
fig_line.update_layout(
    title="Predicted Next Purchase Likelihood per Segment",
    barmode="overlay",
    xaxis_title="Purchase Probability",
    yaxis_title="Number of Customers"
)
st.plotly_chart(fig_line, use_container_width=True)

# ----------------------------------------
# 8. AI-Generated Recommendations
# ----------------------------------------
st.subheader("🤖 AI-Driven Recommendations")
st.write("Personalized actions for each customer segment:")
st.markdown("""
- **High-Value Customers** 💎 → Launch **VIP Loyalty Programs**, early access sales, and premium upsells.
- **Medium-Value Customers** 📈 → Send **personalized product recommendations** to push them into High Value.
- **Low-Value Customers** 🔄 → Use **re-engagement emails, discounts, and ads** to improve retention.
""")

# ----------------------------------------
# 9. Downloadable Report
# ----------------------------------------
csv = filtered_data.to_csv(index=False)
st.download_button(
    label="📥 Download Segmentation Report",
    data=csv,
    file_name="customer_segmentation_report_2026.csv",
    mime="text/csv"
)
