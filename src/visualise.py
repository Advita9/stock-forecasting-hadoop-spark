import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")

plt.figure(figsize=(12,6))

# Plot actual vs predicted
plt.plot(df["label"], label="Actual Price")
plt.plot(df["prediction"], label="Predicted Price")

# Highlight anomalies
anomalies = df[df["is_anomaly"] == 1]
plt.scatter(anomalies.index, anomalies["label"], 
            color="red", label="Anomalies", zorder=5)

plt.title("Stock Price Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(12,5))

plt.plot(df["error"], label="Prediction Error")

plt.axhline(df["error"].mean(), color="green", linestyle="--", label="Mean Error")

plt.title("Prediction Error Over Time")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(12,6))

plt.plot(df["label"], label="Actual", alpha=0.7)

# Mark anomalies clearly
plt.scatter(anomalies.index, anomalies["label"], 
            color="red", s=50, label="Anomaly")

plt.title("Detected Anomalies in Stock Prices")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df["error"], bins=30)

plt.title("Distribution of Prediction Error")
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(12,5))
plt.plot(df["error"], label="Error")
plt.axhline(df["error"].mean() + 2*df["error"].std(), 
            linestyle="--", label="Threshold")
plt.legend()
plt.title("Error vs Threshold")
plt.show()