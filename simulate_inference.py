import time
import requests
import psutil
import pandas as pd
from sklearn.metrics import classification_report

# URL of the FastAPI predict endpoint
PREDICT_URL = "http://localhost:8000/predict"

# Path to the test dataset CSV file 
CSV_PATH = "data/X_test2.csv"

def chunk_data(data, size=100):
    """
    Generator function to yield successive size chunks from data.
    Args:
        data (list): The list of data items to be chunked.
        size (int): The size of each chunk.
    Yields:
        list: A chunk of the input data.
    """
    for i in range(0, len(data), size):
        yield data[i:i + size]

def simulate_edge_inference(delay=0.1):
    # Load the full test dataset into a DataFrame
    df_test = pd.read_csv(CSV_PATH)
    num_samples = len(df_test)
    
    all_predictions = []
    all_latencies = []  
    all_cpu = []       
    all_memory = []    
    
    print(f"Loaded {num_samples} test samples from {CSV_PATH}.")
    print("Starting edge inference simulation...\n")
    
    records = df_test.to_dict(orient="records")
    
    # Process data in chunks
    for chunk in chunk_data(records, size=100):
        # Capture CPU and memory usage before sending the request
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        
        start_time = time.time()
        
        try:
            response = requests.post(PREDICT_URL, json=chunk, timeout=5)
            if response.ok:
                preds = response.json().get("predictions")
                if preds is None:
                    preds = []
            else:
                print(f"Error {response.status_code}: {response.text}")
                preds = []
        except requests.exceptions.Timeout:
            print("Request timed out.")
            preds = []
        except Exception as e:
            print(f"Request failed: {e}")
            preds = []
        
        # Calculate latency for this chunk
        chunk_latency = time.time() - start_time
        
        # Extend the lists: replicate the measured chunk metrics for each prediction in the chunk
        all_latencies.extend([chunk_latency] * len(preds))
        all_cpu.extend([cpu_usage] * len(preds))
        all_memory.extend([memory_usage] * len(preds))
        all_predictions.extend(preds)
        
        print(f"Processed chunk of {len(chunk)} records. Chunk latency: {chunk_latency:.3f} s | CPU: {cpu_usage}% | Memory: {memory_usage}%\n")
        time.sleep(delay)  # Wait for 100ms before sending the next chunk
    
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    avg_cpu = sum(all_cpu) / len(all_cpu) if all_cpu else 0
    avg_memory = sum(all_memory) / len(all_memory) if all_memory else 0
    
    print(f"\n--- Simulation Complete ---")
    print(f"Average Latency per Prediction: {avg_latency:.3f} s")
    print(f"Average CPU Usage per Prediction: {avg_cpu:.1f}%")
    print(f"Average Memory Usage per Prediction: {avg_memory:.1f}%")
    
    df_results = pd.DataFrame({
        "Prediction": all_predictions,
        "Latency": all_latencies,
        "CPU": all_cpu,
        "Memory": all_memory
    })
    df_results.to_csv("data/simulation_results.csv", index=False)
    print("Simulation results saved to data/simulation_results.csv")
    
    y_test = pd.read_csv("data/y_test2.csv")["class"]
    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, all_predictions))

if __name__ == '__main__':
    simulate_edge_inference()
