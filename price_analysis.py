import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'SupplierID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'SupplierName': ['ABC Electronics', 'XYZ Technologies', 'Tech Innovators', 'Global Suppliers', 'Smart Solutions', 'Top Components', 'Gadget Makers', 'Digital Systems', 'Modern Parts', 'Tech Connect'],
    'LeadTime': [5, 7, 3, 6, 4, 8, 5, 6, 4, 7],
    'DeliveryPerformance': [0.9, 0.85, 0.95, 0.88, 0.92, 0.87, 0.93, 0.91, 0.89, 0.84],
    'DefectRate': [0.03, 0.02, 0.01, 0.04, 0.02, 0.03, 0.01, 0.02, 0.04, 0.02],
    'Price': [800, 790, 820, 780, 795, 810, 800, 795, 780, 790]
}

df = pd.DataFrame(data)

X = df[['LeadTime', 'DeliveryPerformance', 'DefectRate']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for predicted_price in y_pred:
    print(f"Predicted Price - ${predicted_price:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.bar(df['SupplierName'], df['LeadTime'])
plt.title('Lead Time')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 3, 2)
plt.bar(df['SupplierName'], df['DeliveryPerformance'])
plt.title('Delivery Performance')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 3, 3)
plt.bar(df['SupplierName'], df['DefectRate'])
plt.title('Defect Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['Price'], df['LeadTime'], label='Lead Time', color='blue')
plt.scatter(df['Price'], df['DeliveryPerformance'], label='Delivery Performance', color='green')
plt.scatter(df['Price'], df['DefectRate'], label='Defect Rate', color='red')

plt.xlabel('Price')
plt.ylabel('Metrics')
plt.title('Predicted Prices vs Supplier Metrics')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['SupplierID'], df['Price'], marker='o', linestyle='-', color='b')

plt.xlabel('Supplier ID')
plt.ylabel('Predicted Price ($)')
plt.title('Predicted Prices vs Supplier IDs')
plt.grid(True)
plt.xticks(df['SupplierID'])
plt.tight_layout()
plt.show()

df = pd.DataFrame(data)

csv_filename = 'supplier_data.csv'
df.to_csv(csv_filename, index=False)