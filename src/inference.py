import pandas as pd  # ✅ REQUIRED
import numpy as np

def make_prediction(df: pd.DataFrame, model):
    try:
        required_features = [
            "pktIAT", "numHdrs", "l4Proto", "ipID", "ack", "seqDiff", "ackDiff", "seqLen", "ackLen", "seqFlowLen",
            "ackFlowLen", "tcpMLen", "tcpMSS", "tcpTmS", "tcpTmER", "tcpOptLen",
            "Time delta from previous captured frame",
            "Time delta from previous displayed frame",
            "Time since first frame", "Time since previous frame",
            "Time since first frame in this TCP stream", "Time since previous frame in this TCP stream",
            "iRTT"
        ]

        available_features = [f for f in required_features if f in df.columns]
        missing_features = [f for f in required_features if f not in df.columns]

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        df = df[available_features]
        processed_array = df.values.astype(np.float32)

        predictions = model.predict(processed_array)
        return predictions
    
    except Exception as e:
        import sys
        print(f"error in make_prediction: {e}", file=sys.stderr, flush=True)
        raise
