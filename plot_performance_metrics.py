import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/simulation_results.csv")
print("CSV Schema:", df.columns.tolist())

# Box Plot for Latency Distribution
plt.figure(figsize=(8, 6))
plt.boxplot(df['Latency'], patch_artist=True,
            boxprops={'facecolor': 'lightblue', 'color': 'blue'},
            medianprops={'color': 'red'})
plt.title("Box Plot of Chunk Latencies")
plt.ylabel("Latency (seconds)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/latency_boxplot.png")
plt.close()

# Histogram for Latency Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Latency'], bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Latency (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram of Chunk Latencies")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/latency_histogram.png")
plt.close()

# Line Chart for CPU Usage Over Time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['CPU'], marker='o', linestyle='-', color='green')
plt.xlabel("Chunk Index")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage Over Time")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/cpu_usage_linechart.png")
plt.close()

# Line Chart for Memory Usage Over Time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Memory'], marker='o', linestyle='-', color='purple')
plt.xlabel("Chunk Index")
plt.ylabel("Memory Usage (%)")
plt.title("Memory Usage Over Time")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/memory_usage_linechart.png")
plt.close()

# Cumulative Distribution Function (CDF) Plot for Latency
latencies = np.sort(df['Latency'].values)
cdf = np.arange(1, len(latencies) + 1) / len(latencies)
plt.figure(figsize=(10, 6))
plt.plot(latencies, cdf, marker='.', linestyle='-', color='orange')
plt.xlabel("Latency (seconds)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Chunk Latencies")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/latency_cdf.png")
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Latency'], marker='o', linestyle='-', color='blue')
plt.title("Chunk Latency Over Time During Edge Inference")
plt.xlabel("Chunk Index")
plt.ylabel("Chunk Latency (seconds)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/chunk_latency_over_time.png")
plt.show()