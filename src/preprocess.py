import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_data(df):
   df = df.copy()

   # convert columns that should be numeric
   numeric_cols = ["pktNo", "pktIAT", "pktTrip", "flowDuration", "ipTTL", "icmpType", "icmpCode", "icmpID", "icmpSeq"]
   for col in numeric_cols:
       if col in df.columns:
           df[col] = pd.to_numeric(df[col], errors="coerce")


   # convert hex values
   hex_cols = ["tcpFlags", "ipHdrChkSum", "ipCalChkSum", "l4HdrChkSum", "l4CalChkSum", "ethType", "ipToS", "ipID"]
   for col in hex_cols:
       if col in df.columns:
           df[col] = df[col].apply(lambda x:
               int(x, 16) if isinstance(x, str) and x.startswith("0x")
               else pd.to_numeric(x, errors="coerce"))


   # encode IP addresses
   if "srcIP" in df.columns:
       le_src = LabelEncoder()
       df["srcIP"] = df["srcIP"].astype(str).fillna("Unknown")
       df["src"] = le_src.fit_transform(df["srcIP"]).astype("int64")
   if "dstIP" in df.columns:
       le_src = LabelEncoder()
       df["dstIP"] = df["dstIP"].astype(str).fillna("Unknown")
       df["dstIP"] = le_src.fit_transform(df["dstIP"]).astype("int64")

   # normalize numerical features
   scaler = MinMaxScaler()
   norm_cols = ["pktLen", "srcPort", "dstPort"]
   for col in norm_cols:
       if col in df.columns:
           df[col] = pd.to_numeric(df[col], errors="coerce")


   # apply scaler only if norm_cols exist in the dataframe
   norm_cols_exist = [col for col in norm_cols if col in df.columns]
   if norm_cols_exist:
       # First fill NaNs in these columns before scaling
       for col in norm_cols_exist:
           df[col].fillna(df[col].median(), inplace=True)
       df[norm_cols_exist] = scaler.fit_transform(df[norm_cols_exist])


   # convert all object columns to numeric
   object_columns = df.select_dtypes(include=['object']).columns
   for col in object_columns:
       converted = pd.to_numeric(df[col], errors='coerce')
       if converted.isna().mean() > 0.5:
           le = LabelEncoder()
           df[col] = df[col].fillna('unknown')
           df[col] = le.fit_transform(df[col].astype(str))
       else:
           df[col] = converted


   # fill missing values column by column using the median
   numeric_columns = df.select_dtypes(include=['number']).columns
   for col in numeric_columns:
       df[col] = df[col].fillna(df[col].median())


   # fill any remaining NaNs with 0
   df.fillna(0, inplace=True)


   # verify no NaNs remain
   assert not df.isna().any().any(), "NaN values still exist in the DataFrame"


   return df
