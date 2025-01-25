###cc###

import numpy as np

# Data
refineries = ["Greece", "Poland", "Spain", "UK"]
products = ["Gasoline-87", "Gasoline-89", "Gasoline-92", "Jet Fuel", "Diesel Fuel", "Heating Oil"]
capacities = np.array([400000, 540000, 625000, 735000])
demands = np.array([
    [35000, 45000, 50000, 20000, 75000, 25000],  # Greece
    [22000, 38000, 60000, 25000, 35000, 205000], # Poland
    [76000, 103000, 83000, 47000, 125000, 30000], # Spain
    [98000, 52000, 223000, 127000, 87000, 13000]  # UK
])
sales_prices = np.array([90.45, 93.66, 95.50, 61.25, 101.64, 66.36])

processing_cost = 19  # $/barrel
crude_oil_cost = 57  # $/barrel
discount_rate = 0.07  # Discount for excess sales

# Transportation costs
port_cost_ceyhan = 124000  # Fixed port cost ($)
ship_rental_cost = 33000  # $/day
fuel_cost_per_hour = 3000  # $/hour
travel_times = np.array([2, 15, 8, 12])  # Travel times in days for each refinery

# Decision Variables
production = demands.copy()
excess_sales = np.zeros_like(demands)  # Initialize excess sales

# Adjust production to fill refinery capacity
for i in range(len(refineries)):
    available_capacity = capacities[i] - production[i].sum()
    for j in range(len(products)):
        if available_capacity > 0:
            additional_production = min(available_capacity, production[i][j])
            excess_sales[i][j] += additional_production
            available_capacity -= additional_production

# Revenue from production
revenue_production = (production * sales_prices).sum()

# Revenue from excess sales
revenue_excess_sales = ((1 - discount_rate) * excess_sales * sales_prices).sum()

# Total revenue
total_revenue = revenue_production + revenue_excess_sales

# Crude oil cost
total_crude_oil_cost = (production.sum() + excess_sales.sum()) * crude_oil_cost

# Processing cost
total_processing_cost = (production.sum() + excess_sales.sum()) * processing_cost

# Transportation cost
transport_cost = port_cost_ceyhan
for i, travel_time in enumerate(travel_times):
    transport_cost += (
        ship_rental_cost * travel_time +
        fuel_cost_per_hour * travel_time * 24
    )

# Total costs
total_costs = total_crude_oil_cost + total_processing_cost + transport_cost

# Profit
profit = total_revenue - total_costs

# Results
print("Revenue from Production ($):", revenue_production)
print("Revenue from Excess Sales ($):", revenue_excess_sales)
print("Total Revenue ($):", total_revenue)
print("Crude Oil Cost ($):", total_crude_oil_cost)
print("Processing Cost ($):", total_processing_cost)
print("Transportation Cost ($):", transport_cost)
print("Total Costs ($):", total_costs)
print("Profit ($):", profit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sabit Veriler
refinery_demands_min = {
    "Greece": {
        "Gasoline-87": 35000,
        "Gasoline-89": 45000,
        "Gasoline-92": 50000,
        "Jet Fuel": 20000,
        "Diesel Fuel": 75000,
        "Heating Oil": 25000,
    },
    "Poland": {
        "Gasoline-87": 22000,
        "Gasoline-89": 38000,
        "Gasoline-92": 60000,
        "Jet Fuel": 25000,
        "Diesel Fuel": 35000,
        "Heating Oil": 205000,
    },
    "Spain": {
        "Gasoline-87": 76000,
        "Gasoline-89": 103000,
        "Gasoline-92": 83000,
        "Jet Fuel": 47000,
        "Diesel Fuel": 125000,
        "Heating Oil": 30000,
    },
    "UK": {
        "Gasoline-87": 98000,
        "Gasoline-89": 52000,
        "Gasoline-92": 223000,
        "Jet Fuel": 127000,
        "Diesel Fuel": 87000,
        "Heating Oil": 13000,
    },
}

refinery_capacities = {
    "Greece": 400000,
    "Poland": 540000,
    "Spain": 625000,
    "UK": 735000,
}

product_prices = {
    "Gasoline-87": 90.45,
    "Gasoline-89": 93.66,
    "Gasoline-92": 95.50,
    "Jet Fuel": 61.25,
    "Diesel Fuel": 101.64,
    "Heating Oil": 66.36,
}

discount_rate = 0.07  # Fazla satışlarda %7 indirim
discounted_prices = {
    product: round(price * (1 - discount_rate), 2) for product, price in product_prices.items()
}

# Maliyet Verileri
crude_price_per_barrel = 57  # Ham petrol fiyatı ($/varil)
refinery_cost_per_barrel = 19  # Rafineri üretim maliyeti ($/varil)
fuel_cost_per_hour = 3000  # Saatlik yakıt maliyeti ($)
port_charge = 124000  # Liman masrafı ($)
daily_ship_rental = 33000  # Geminin günlük kira bedeli ($)
max_ship_capacity = 645000  # Maksimum taşınabilir kapasite (varil)

# Seyahat Süreleri (Gün)
travel_days = {
    "Greece": 2,
    "Poland": 15,
    "Spain": 8,
    "UK": 12,
}

# Hesaplama Fonksiyonu
def calculate_refinery_finances(refinery_name, demands, capacity, travel_days):
    """
    Bir rafineri için toplam gelir, masraf, net kar ve satılan yakıt miktarlarını hesaplama.
    """
    # Maksimum satılabilir kapasiteyi hesapla
    sellable_capacity = min(capacity, max_ship_capacity)

    # Masraflar
    crude_cost = sellable_capacity * crude_price_per_barrel
    refinery_cost = sellable_capacity * refinery_cost_per_barrel

    travel_hours = travel_days * 24
    fuel_cost = travel_hours * fuel_cost_per_hour
    ship_rental_cost = travel_days * daily_ship_rental
    total_transport_cost = fuel_cost + port_charge + ship_rental_cost

    # Toplam Masraf
    total_expense = crude_cost + refinery_cost + total_transport_cost

    # Gelirler
    product_revenue = sum(demand * product_prices[product] for product, demand in demands.items())

    # Satılan yakıt detayları
    fuel_sales = {product: {"Demand": demand, "Revenue": demand * product_prices[product], "Discounted": 0} for product, demand in demands.items()}

    # Fazladan kapasite kontrolü
    total_demand = sum(demands.values())
    excess_capacity = max(0, sellable_capacity - total_demand)

    excess_sales_revenue = 0
    excess_fuel_sold = 0
    if excess_capacity > 0:
        diesel_price = discounted_prices["Diesel Fuel"]
        excess_fuel_sold = excess_capacity
        excess_sales_revenue = excess_capacity * diesel_price
        if "Diesel Fuel" in fuel_sales:
            fuel_sales["Diesel Fuel"]["Demand"] += excess_capacity
            fuel_sales["Diesel Fuel"]["Revenue"] += excess_sales_revenue
            fuel_sales["Diesel Fuel"]["Discounted"] += excess_capacity
        else:
            fuel_sales["Diesel Fuel"] = {"Demand": excess_capacity, "Revenue": excess_sales_revenue, "Discounted": excess_capacity}

    # Toplam Gelir
    total_revenue = product_revenue + excess_sales_revenue

    # Net Kâr
    net_profit = total_revenue - total_expense

    return {
        "Refinery": refinery_name,
        "Sellable Capacity": sellable_capacity,
        "Total Revenue ($)": total_revenue,
        "Total Expense ($)": total_expense,
        "Net Profit ($)": net_profit,
        "Crude Cost ($)": crude_cost,
        "Refinery Cost ($)": refinery_cost,
        "Transport Cost ($)": total_transport_cost,
        "Excess Sales Revenue ($)": excess_sales_revenue,
        "Fuel Sales": fuel_sales,
    }

# Bütün Rafineriler için Hesaplama
results = []
for refinery, demands in refinery_demands_min.items():
    result = calculate_refinery_finances(
        refinery_name=refinery,
        demands=demands,
        capacity=refinery_capacities[refinery],
        travel_days=travel_days[refinery]
    )
    results.append(result)

# Sonuçları Pandas DataFrame ile Gösterme
summary = pd.DataFrame([{key: value for key, value in result.items() if key != "Fuel Sales"} for result in results])

# Satılan yakıt detaylarını ekrana yazdır
print("Satılan Yakıt Detayları:")
for result in results:
    print(f"--- {result['Refinery']} ---")
    for fuel, details in result["Fuel Sales"].items():
        print(f"{fuel}: {details['Demand']} varil, Gelir: ${details['Revenue']:.2f}, İndirimli Satış: {details['Discounted']} varil")

# Sonuçları Yazdır
print("\nGenel Finansal Özet:")
print(summary)

# Genel Toplam Kâr
total_profit = summary["Net Profit ($)"].sum()
print(f"\nToplam Net Kâr: ${total_profit:,.2f}")

# Sabit Veriler
import pandas as pd

refinery_demands_min = {
    "Greece": {
        "Gasoline-87": 35000,
        "Gasoline-89": 45000,
        "Gasoline-92": 50000,
        "Jet Fuel": 20000,
        "Diesel Fuel": 75000,
        "Heating Oil": 25000,
    },
    "Poland": {
        "Gasoline-87": 22000,
        "Gasoline-89": 38000,
        "Gasoline-92": 60000,
        "Jet Fuel": 25000,
        "Diesel Fuel": 35000,
        "Heating Oil": 205000,
    },
    "Spain": {
        "Gasoline-87": 76000,
        "Gasoline-89": 103000,
        "Gasoline-92": 83000,
        "Jet Fuel": 47000,
        "Diesel Fuel": 125000,
        "Heating Oil": 30000,
    },
    "UK": {
        "Gasoline-87": 98000,
        "Gasoline-89": 52000,
        "Gasoline-92": 223000,
        "Jet Fuel": 127000,
        "Diesel Fuel": 87000,
        "Heating Oil": 13000,
    },
}

refinery_capacities = {
    "Greece": 400000,
    "Poland": 540000,
    "Spain": 625000,
    "UK": 735000,
}

product_prices = {
    "Gasoline-87": 90.45,
    "Gasoline-89": 93.66,
    "Gasoline-92": 95.50,
    "Jet Fuel": 61.25,
    "Diesel Fuel": 101.64,
    "Heating Oil": 66.36,
}

discount_rate = 0.07
discounted_prices = {
    product: round(price * (1 - discount_rate), 2) for product, price in product_prices.items()
}

# Maliyet Verileri
crude_price_per_barrel = 57
refinery_cost_per_barrel = 19
fuel_cost_per_hour = 3000
port_charge = 124000
daily_ship_rental = 33000
max_ship_capacity = 645000

# Seyahat Süreleri (Gün)
travel_days = {
    "Greece": 2,
    "Poland": 15,
    "Spain": 8,
    "UK": 12,
}

# Hesaplama Fonksiyonu
def calculate_refinery_finances(refinery_name, demands, capacity, travel_days):
    sellable_capacity = min(capacity, max_ship_capacity)

    crude_cost = sellable_capacity * crude_price_per_barrel
    refinery_cost = sellable_capacity * refinery_cost_per_barrel

    travel_hours = travel_days * 24
    fuel_cost = travel_hours * fuel_cost_per_hour
    ship_rental_cost = travel_days * daily_ship_rental
    total_transport_cost = fuel_cost + port_charge + ship_rental_cost

    total_expense = crude_cost + refinery_cost + total_transport_cost

    product_revenue = sum(demand * product_prices[product] for product, demand in demands.items())

    fuel_sales = {product: {"Demand": demand, "Revenue": demand * product_prices[product], "Discounted": 0} for product, demand in demands.items()}

    total_demand = sum(demands.values())
    excess_capacity = max(0, sellable_capacity - total_demand)

    excess_sales_revenue = 0
    if excess_capacity > 0:
        diesel_price = discounted_prices["Diesel Fuel"]
        fuel_sales["Diesel Fuel"]["Demand"] += excess_capacity
        fuel_sales["Diesel Fuel"]["Revenue"] += excess_capacity * diesel_price
        fuel_sales["Diesel Fuel"]["Discounted"] += excess_capacity
        excess_sales_revenue = excess_capacity * diesel_price

    total_revenue = product_revenue + excess_sales_revenue
    net_profit = total_revenue - total_expense

    return {
        "Refinery": refinery_name,
        "Sellable Capacity": sellable_capacity,
        "Total Revenue ($)": total_revenue,
        "Total Expense ($)": total_expense,
        "Net Profit ($)": net_profit,
        "Crude Cost ($)": crude_cost,
        "Refinery Cost ($)": refinery_cost,
        "Transport Cost ($)": total_transport_cost,
        "Excess Sales Revenue ($)": excess_sales_revenue,
        "Fuel Sales": fuel_sales,
    }

# Rafineriler için hesaplama
results = []
for refinery, demands in refinery_demands_min.items():
    result = calculate_refinery_finances(
        refinery_name=refinery,
        demands=demands,
        capacity=refinery_capacities[refinery],
        travel_days=travel_days[refinery]
    )
    results.append(result)

# Yakıt Satışları Tablosu
fuel_sales_data = []
for result in results:
    for fuel, details in result["Fuel Sales"].items():
        fuel_sales_data.append({
            "Refinery": result["Refinery"],
            "Fuel Type": fuel,
            "Total Sold (barrels)": details["Demand"],
            "Revenue ($)": details["Revenue"],
            "Discounted Sales (barrels)": details["Discounted"]
        })

fuel_sales_df = pd.DataFrame(fuel_sales_data)

# Finansal Özet Tablosu
summary_df = pd.DataFrame([{key: value for key, value in result.items() if key != "Fuel Sales"} for result in results])

# Çıktılar
print("Satılan Yakıt Detayları Tablosu:")
print(fuel_sales_df)

print("\nGenel Finansal Özet Tablosu:")
print(summary_df)

total_profit = summary_df["Net Profit ($)"].sum()
print(f"\nToplam Net Kâr: ${total_profit:,.2f}")

summary_df.head(20)

fuel_sales_df.head(25)

# Seyahat sürelerini eklemek için "Transport Days" sütununu oluşturma
travel_days = {"Greece": 2, "Poland": 15, "Spain": 8, "UK": 12}
summary_df["Transport Days"] = summary_df["Refinery"].map(travel_days)

# Güncellenmiş DataFrame'i göster
print(summary_df)

import seaborn as sns


plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Refinery", y="Net Profit ($)", palette="coolwarm")
plt.title("Rafinerilere Göre Net Kar")
plt.xlabel("Rafineri")
plt.ylabel("Net Kar ($)")
plt.show()

# Taşıma Süresinin Net Kara Oranı
summary_df["Transport to Net Profit Ratio"] = summary_df["Net Profit ($)"] / summary_df["Transport Cost ($)"]
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Refinery", y="Transport to Net Profit Ratio", palette="viridis")
plt.title("Net Kar / Taşıma Masrafları")
plt.xlabel("Rafineri")
plt.ylabel("Taşıma Masraflarının Net Kara Oranı")
plt.show()

# Fazladan Satış Gelirlerinin Net Kara Etkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=summary_df, x="Excess Sales Revenue ($)", y="Net Profit ($)", hue="Refinery", style="Refinery", s=150)
plt.title("Fazladan Satış Gelirlerinin Net Kara Etkisi")
plt.xlabel("Fazladan Satış Geliri ($)")
plt.ylabel("Net Kar ($)")
plt.legend(title="Rafineri")
plt.show()

# Taşıma Günlerine Göre Toplam Gelir
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Transport Days", y="Total Revenue ($)", hue="Refinery", palette="pastel")
plt.title("Taşıma Günlerine Göre Toplam Gelir")
plt.xlabel("Taşıma Günleri")
plt.ylabel("Toplam Gelir ($)")
plt.legend(title="Rafineri")
plt.show()

import seaborn as sns

# Set a professional style for the plots
sns.set_theme(style="whitegrid")

# Plot 1: Total Revenue vs Total Expense for each refinery
plt.figure(figsize=(10, 6))
df_melted = df.melt(id_vars="Refinery", value_vars=["Total Revenue ($)", "Total Expense ($)"],
                    var_name="Category", value_name="Amount")
sns.barplot(x="Refinery", y="Amount", hue="Category", data=df_melted, palette="Blues_d")
plt.title("Total Revenue vs Total Expense by Refinery", fontsize=16)
plt.xlabel("Refinery", fontsize=12)
plt.ylabel("Amount ($)", fontsize=12)
plt.legend(title="Category")
plt.tight_layout()
plt.show()

# Plot 2: Net Profit for each refinery
plt.figure(figsize=(10, 6))
sns.barplot(x="Refinery", y="Net Profit ($)", data=df, color="green", saturation=0.8)
plt.title("Net Profit by Refinery", fontsize=16)
plt.xlabel("Refinery", fontsize=12)
plt.ylabel("Net Profit ($)", fontsize=12)
plt.tight_layout()
plt.show()

# Plot 3: Excess Sales Revenue for each refinery
plt.figure(figsize=(10, 6))
sns.barplot(x="Refinery", y="Excess Sales Revenue ($)", data=df, color="orange", saturation=0.8)
plt.title("Excess Sales Revenue by Refinery", fontsize=16)
plt.xlabel("Refinery", fontsize=12)
plt.ylabel("Excess Sales Revenue ($)", fontsize=12)
plt.tight_layout()
plt.show()

# Import libraries
import pandas as pd

# Constants
crude_price_per_barrel = 71  # \$/varil
refinery_cost_per_barrel = 19  # \$/varil
fuel_cost_per_hour = 3000  # Saatlik yakıt maliyeti (\$/saat)
melkøya_port_charge = 156000  # Liman masrafı (\$)
daily_ship_rental = 33000  # Gemi günlük kirası (\$/gün)
discount_rate = 0.07

# Product prices
product_prices = {
    "Gasoline-87": 90.45,
    "Gasoline-89": 93.66,
    "Gasoline-92": 95.50,
    "Jet Fuel": 61.25,
    "Diesel Fuel": 101.64,
    "Heating Oil": 66.36,
}

# Refinery capacities
refinery_capacities = {
    "Greece": 400000,
    "Poland": 540000,
    "Spain": 625000,
    "UK": 735000
}

# Refinery demands
refinery_demands_min = {
    "Greece": {
        "Gasoline-87": 35000,
        "Gasoline-89": 45000,
        "Gasoline-92": 50000,
        "Jet Fuel": 20000,
        "Diesel Fuel": 75000,
        "Heating Oil": 25000,
    },
    "Poland": {
        "Gasoline-87": 22000,
        "Gasoline-89": 38000,
        "Gasoline-92": 60000,
        "Jet Fuel": 25000,
        "Diesel Fuel": 35000,
        "Heating Oil": 205000,
    },
    "Spain": {
        "Gasoline-87": 76000,
        "Gasoline-89": 103000,
        "Gasoline-92": 83000,
        "Jet Fuel": 47000,
        "Diesel Fuel": 125000,
        "Heating Oil": 30000,
    },
    "UK": {
        "Gasoline-87": 98000,
        "Gasoline-89": 52000,
        "Gasoline-92": 223000,
        "Jet Fuel": 127000,
        "Diesel Fuel": 87000,
        "Heating Oil": 13000,
    },
}

# Travel days from Melkøya
travel_days_melkøya = {
    "Greece": 11,
    "Poland": 3,
    "Spain": 4,
    "UK": 3,
}

def calculate_with_max_capacity(refinery_name, demands, capacity, travel_days):
    """
    Calculate finances assuming full refinery capacity is utilized for costs.
    """
    discounted_diesel_price = product_prices["Diesel Fuel"] * (1 - discount_rate)

    # Calculate costs based on max capacity
    crude_cost = capacity * crude_price_per_barrel
    refinery_cost = capacity * refinery_cost_per_barrel

    # Calculate actual revenue based on demands
    product_revenue = 0
    product_details = []
    total_demand = sum(demands.values())
    processed_demand = min(total_demand, capacity)
    excess_capacity = max(0, capacity - processed_demand)

    for product, demand in demands.items():
        processed_product_demand = min(demand, processed_demand)
        processed_demand -= processed_product_demand
        revenue = processed_product_demand * product_prices[product]
        product_revenue += revenue

        product_details.append({
            "Product": product,
            "Demand Processed": processed_product_demand,
            "Revenue ($)": revenue,
        })

    # Excess capacity revenue (diesel fuel)
    excess_sales_revenue = excess_capacity * discounted_diesel_price

    # Transportation and port costs
    travel_hours = travel_days * 24
    fuel_cost = travel_hours * fuel_cost_per_hour
    ship_rental_cost = travel_days * daily_ship_rental

    # Total revenue and costs
    total_revenue = product_revenue + excess_sales_revenue
    total_transport_cost = fuel_cost + ship_rental_cost + melkøya_port_charge
    total_cost = crude_cost + refinery_cost + total_transport_cost
    net_profit = total_revenue - total_cost

    return {
        "Refinery": refinery_name,
        "Total Revenue ($)": total_revenue,
        "Total Expense ($)": total_cost,
        "Net Profit ($)": net_profit,
        "Crude Cost ($)": crude_cost,
        "Refinery Cost ($)": refinery_cost,
        "Transport Cost ($)": total_transport_cost,
        "Excess Sales Revenue ($)": excess_sales_revenue,
        "Details": product_details
    }

# Calculate for all refineries excluding Poland
results_no_poland = []
for refinery, demands in refinery_demands_min.items():
    if refinery == "Poland":
        continue  # Skip Poland
    result = calculate_with_max_capacity(
        refinery_name=refinery,
        demands=demands,
        capacity=refinery_capacities[refinery],  # Use refinery capacities from the document
        travel_days=travel_days_melkøya[refinery]
    )
    results_no_poland.append(result)

# Create a DataFrame to summarize the results
summary_no_poland = pd.DataFrame([
    {
        "Refinery": result["Refinery"],
        "Total Revenue ($)": result["Total Revenue ($)"],
        "Total Expense ($)": result["Total Expense ($)"],
        "Net Profit ($)": result["Net Profit ($)"],
        "Crude Cost ($)": result["Crude Cost ($)"],
        "Refinery Cost ($)": result["Refinery Cost ($)"],
        "Transport Cost ($)": result["Transport Cost ($)"],
        "Excess Sales Revenue ($)": result["Excess Sales Revenue ($)"],
    }
    for result in results_no_poland
])

# Display financial summary and total operation profit
total_operation_profit = summary_no_poland["Net Profit ($)"].sum()
print("=== Refinery Financial Summary (Excluding Poland) ===")
print(summary_no_poland)
print("\n=== Total Operation Profit ===")
print(f"Total Profit (Net): ${total_operation_profit:,.2f}")

# Korelasyon matrisi oluşturma ve görselleştirme (summary_df için)
corr_matrix_summary = summary_df.drop(columns=["Refinery"]).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_summary, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Korelasyon Matrisi")
plt.show()