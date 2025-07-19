import streamlit as st
import pandas as pd
from fpdf import FPDF
import io
from datetime import date

st.set_page_config(page_title="Prep and Prime GH Inventory Dashboard", layout="wide")
st.title("Prep and Prime GH Inventory Dashboard")

# Google Sheets public CSV export links
INVENTORY_CSV = "https://docs.google.com/spreadsheets/d/1aiukBw220yTYGQjvrb1C0Tc_AsYM4p8EwJ70CWf1vhU/export?format=csv&id=1aiukBw220yTYGQjvrb1C0Tc_AsYM4p8EwJ70CWf1vhU&gid=0"
CUSTOMERS_CSV = "https://docs.google.com/spreadsheets/d/1by2BJB3H40vlggY31TNM-FS7H5TO5z9rQHrDE6lXPX0/export?format=csv&id=1by2BJB3H40vlggY31TNM-FS7H5TO5z9rQHrDE6lXPX0&gid=0"
SALES_CSV = "https://docs.google.com/spreadsheets/d/15qSz8hweToyDIeulHBTUN5dVBNw6tZojC9CxupTgWRk/export?format=csv&id=15qSz8hweToyDIeulHBTUN5dVBNw6tZojC9CxupTgWRk&gid=0"

@st.cache_data(ttl=60)
def get_data(url):
    return pd.read_csv(url)

inventory = get_data(INVENTORY_CSV)
customers = get_data(CUSTOMERS_CSV)
sales = get_data(SALES_CSV)

tabs = st.tabs([
    "Dashboard",
    "Inventory List",
    "All Customers",
    "Receipt Generator"
])

# Dashboard
with tabs[0]:
    st.header("Dashboard")
    st.metric("Total Products", len(inventory))
    st.metric("Low Stock (< 5)", (inventory['Quantity'] < 5).sum())
    st.metric("Total Customers", len(customers))
    st.metric("Total Sales (Receipts)", len(sales['Receipt No'].unique()) if not sales.empty else 0)
    st.subheader("Low Stock Products")
    st.dataframe(inventory[inventory['Quantity'] < 5])

# Inventory List
with tabs[1]:
    st.header("Inventory List")
    search = st.text_input("Search by product, brand or category")
    if search:
        mask = inventory.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        st.dataframe(inventory[mask])
    else:
        st.dataframe(inventory)

# All Customers
with tabs[2]:
    st.header("All Customers")
    cust_search = st.text_input("Search by customer name, phone, or location")
    if cust_search:
        mask = customers.apply(lambda row: cust_search.lower() in str(row).lower(), axis=1)
        st.dataframe(customers[mask])
    else:
        st.dataframe(customers)

# Receipt Generator
def generate_receipt_pdf(customer, items, total, receipt_no, date_str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Prep and Prime GH", ln=True, align="C")
    pdf.cell(200, 10, "RECEIPT", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(100, 10, f"Date: {date_str}", ln=True)
    pdf.cell(100, 10, f"Receipt No: {receipt_no}", ln=True)
    pdf.cell(100, 10, f"Customer: {customer}", ln=True)
    pdf.ln(10)
    pdf.cell(70, 10, "Product", 1)
    pdf.cell(30, 10, "Qty", 1)
    pdf.cell(40, 10, "Price", 1)
    pdf.cell(40, 10, "Total", 1)
    pdf.ln()
    for item in items:
        pdf.cell(70, 10, str(item['product']), 1)
        pdf.cell(30, 10, str(item['qty']), 1)
        pdf.cell(40, 10, f"{item['price']:.2f}", 1)
        pdf.cell(40, 10, f"{item['total']:.2f}", 1)
        pdf.ln()
    pdf.ln(5)
    pdf.cell(140, 10, "Grand Total", 1)
    pdf.cell(40, 10, f"{total:.2f}", 1)
    out = io.BytesIO()
    pdf.output(out)
    return out.getvalue()

with tabs[3]:
    st.header("Receipt Generator")
    customer_options = customers["Name"].drop_duplicates().tolist()
    inventory_options = inventory["Product Name"].drop_duplicates().tolist()

    with st.form("receipt_form"):
        col1, col2 = st.columns(2)
        with col1:
            customer = st.selectbox("Select Customer", customer_options)
            receipt_no = st.text_input("Receipt No", f"REC-{date.today().strftime('%Y%m%d')}-001")
            date_str = st.date_input("Date", value=date.today())
        with col2:
            product_selections = st.multiselect("Select Products", inventory_options)
        items = []
        total = 0
        for prod in product_selections:
            prod_row = inventory[inventory['Product Name'] == prod].iloc[0]
            qty = st.number_input(f"Quantity for {prod}", min_value=1, max_value=int(prod_row['Quantity']), step=1, key=prod)
            price = prod_row['Price']
            subtotal = qty * price
            items.append({'product': prod, 'qty': qty, 'price': price, 'total': subtotal})
            total += subtotal

        st.write(f"**Grand Total:** GHS {total:.2f}")
        submitted = st.form_submit_button("Generate Receipt PDF")
        if submitted and customer and items:
            pdf_bytes = generate_receipt_pdf(customer, items, total, receipt_no, str(date_str))
            st.success("Receipt generated!")
            st.download_button("Download Receipt", data=pdf_bytes, file_name=f"{receipt_no}.pdf", mime="application/pdf")

st.caption("All editing is managed in Google Sheets. This dashboard is for viewing, search, and receipt generation only.")
