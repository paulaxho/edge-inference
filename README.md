
# Edge Inference for IoV – Real-Time DoS Detection

This project is a lightweight, containerised machine learning system that detects Denial-of-Service (DoS) attacks in Internet of Vehicles (IoV) environments using an XGBoost model served via FastAPI.

It simulates edge deployment with strict resource limits (1 CPU, 1GB RAM).


How to run 

1. Clone the Repository:

   git clone https://github.com/paulaxho/iov-edge-detector.git
   cd iov-edge-detector

2. Start the API in Docker:
  - you will need to have Docker Desktop installed in your device 

    docker-compose down
    docker-compose up --build


   This will:
   - Build a Docker image
   - Run the FastAPI app
   - Simulate an edge device with 1 CPU and 1GB memory

   The API will be live at (use Google Chrome):
   http://localhost:8000

How to test it 

Option A – Upload a CSV:
- Open http://localhost:8000 in your browser
- Upload a .csv file (e.g., X_test2.csv)
- You'll see a table of predictions


SIMULATE REAL-TIME INFERENCE

To simulate real-time input and track latency, CPU, and memory, run this line in the terminal:

   python simulate_edge_inference.py

It sends test data in chunks, logs latency per prediction, and saves the results to:
   data/simulation_results.csv


After running the simulation:

   python plot_performance_metrics.py

This generates:
- latency_boxplot.png
- cpu_usage_linechart.png
- memory_usage_linechart.png

All saved in the results/ folder.