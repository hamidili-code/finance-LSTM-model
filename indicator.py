import os
import sys
import pandas as pd
import talib

DATA_FOLDER = "csv" 

symbol = input("enter your symbol: ").strip().upper()
timeframe = input("enter your timeframe: ").strip().lower()

if not symbol:
    print("❌ no symbol provided")
    sys.exit()

safe_name = symbol.replace('/', '_').replace('=', '').replace(' ', '_')

input_file_name = f"{safe_name}_{timeframe}.xlsx" 
input_file_path = os.path.join(DATA_FOLDER, input_file_name)

output_file_name = f"{safe_name}_TA_Indicators_Output.csv"
output_file_path = os.path.join(DATA_FOLDER, output_file_name)


try:
    print(f"\n⌛️ loading data from file: {input_file_path}")
    
    df = pd.read_excel(input_file_path, index_col=0)
    
    if 'Volume' not in df.columns:
        print("⚠️ Warning: 'Volume' column not found. MFI will use zeros.")
        df['Volume'] = 0.0
        
    open_price = df['Open'].values.astype('float64')
    close = df['Close'].values.astype('float64')
    high = df['High'].values.astype('float64')
    low = df['Low'].values.astype('float64')

    
    print(f"✅ {len(df)} data rows loaded successfully.")

except FileNotFoundError:
    print(f"❌ Error: File {input_file_name} not found in folder '{DATA_FOLDER}'.")
    print("❌ Please ensure you have saved the raw data using the download code.")
    sys.exit()
except Exception as e:
    print(f"❌ Error loading file: {e}")
    sys.exit()


print("⚙️ Calculating technical indicators...")

df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(
    close, fastperiod=12, slowperiod=26, signalperiod=9
)

df['RSI'] = talib.RSI(close, timeperiod=14)

df['EMA_9'] = talib.EMA(close, timeperiod=9)
df['EMA_50'] = talib.EMA(close, timeperiod=50)
df['EMA_100'] = talib.EMA(close, timeperiod=100)


df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
    high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
)
df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

df['MOM'] = talib.MOM(close, timeperiod=10)
 

df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(
    close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0 
)
df['SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)

print("✅ calculation of technical indicators completed.")


try:
    df_final = df.dropna()

    df_final.to_csv(output_file_path, index=True)
    print(f"✅ Final data (including indicators) saved to **{output_file_path}**.")

except Exception as e:
    print(f"❌ Error saving file: {e}")

print("\n--- Process completed. ---")