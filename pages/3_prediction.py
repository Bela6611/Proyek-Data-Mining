import streamlit as st
import pandas as pd

st.set_page_config(page_title="Prediksi Penjualan", page_icon="🧮")
st.header("🧮 Prediksi Penjualan Sederhana (Rule-Based)")

# Input data manual oleh user
st.subheader("Masukkan Data Transaksi")

price = st.number_input("Harga Barang (price)", min_value=0.0, value=100.0)
quantity = st.number_input("Jumlah (quantity)", min_value=1, value=1)
cost = st.number_input("Total Biaya (cost)", min_value=0.0, value=50.0)
profit = st.number_input("Keuntungan (profit)", min_value=0.0, value=20.0)

# Prediksi sederhana: sales = price * quantity + profit - cost
if st.button("Prediksi Penjualan"):
    sales = price * quantity + profit - cost
    st.success(f"💰 Prediksi Penjualan: **${sales:,.2f}**")
