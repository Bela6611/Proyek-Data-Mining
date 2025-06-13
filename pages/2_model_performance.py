import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Evaluasi Model", page_icon="📊")
st.header("📊 Evaluasi Penjualan Sederhana")

@st.cache_data
def load_data():
    df = pd.read_csv("SuperStore_Sales_Updated (1).csv")
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
    df = df.dropna()
    return df

df = load_data()

# Prediksi manual: sales = price * quantity + profit - cost
df['predicted_sales'] = df['price'] * df['quantity'] + df['profit'] - df['cost']
df['error'] = df['sales'] - df['predicted_sales']

# Ringkasan evaluasi
st.subheader("📋 Ringkasan Evaluasi (Manual)")
st.write(f"Jumlah Data: {len(df)}")
st.write(f"Total Error Absolut: {df['error'].abs().sum():,.2f}")
st.write(f"Rata-rata Error Absolut: {df['error'].abs().mean():,.2f}")

# Grafik
st.subheader("📈 Grafik Prediksi vs Aktual (sampel 100 data pertama)")
sample = df.head(100)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sample['sales'].values, label='Aktual', marker='o')
ax.plot(sample['predicted_sales'].values, label='Prediksi', marker='x')
ax.set_xlabel("Index")
ax.set_ylabel("Penjualan")
ax.legend()
st.pyplot(fig)
