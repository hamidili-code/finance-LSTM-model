import yfinance as yf
import os
import sys
import openpyxl

DATA_FOLDER = "csv" 

symbol = input("enter your symbol(like EURUSD=X) ").strip().upper()

if not symbol:
    print("❌ no symbol provided. Exiting.")
    sys.exit()

timeframe = input("enter your timeframe (1m, ...) ").strip().lower()

if not timeframe:
    print("❌ no timeframe provided. Exiting.")
    sys.exit()

print(f"\n--- star for {symbol} ---")

try:
    os.makedirs(DATA_FOLDER, exist_ok=True)
    print(f"✅ files in folder '{DATA_FOLDER}' will be saved.")
except Exception as e:
    print(f"❌ error in folder '{DATA_FOLDER}': {e}")
    sys.exit()

try:
    print(f"⌛️ download data for {symbol} ...")
    

    data = yf.download(tickers=symbol, 
                       period="10y", 
                       interval=timeframe,
                       progress=False)

    if data.empty:
        print(f"❌ data for {symbol} not found. Exiting.")
        sys.exit()
        
    required_data = data[['Open', 'High', 'Low', 'Close']].copy()
    required_data.dropna(inplace=True) 
    
    if required_data.empty:
        print(f"--- no data for {symbol} not found. ---")
        sys.exit()

    print(f"✅ {len(required_data)} data rows downloaded for {symbol}.")

except Exception as e:
    print(f"❌ error in download {symbol} accured: {e}")
    sys.exit()
    

try:
    safe_name = symbol.replace('/', '_').replace('=', '').replace(' ', '_')
    file_name = f"{safe_name}_{timeframe}.xlsx"
    file_path = os.path.join(DATA_FOLDER, file_name)
    
    required_data.to_excel(file_path, index=True, header=True)
except Exception as e:
    print(f"❌ Error downloading or saving symbol {symbol}: {e}")

try:
            wb = openpyxl.load_workbook(file_path)
            ws = wb.active
            

            ws.delete_rows(3, 1) 
            
            ws.delete_rows(2, 1) 

            wb.save(file_path)
            print("   ✅ rows with missing data removed successfully.")
            
except ImportError:
            print("   ⚠️ Warning: 'openpyxl' library is required to remove extra rows. Please install it.")
except Exception as openpyxl_error:
            print(f"   ❌ Error removing rows with openpyxl: {openpyxl_error}")

print(f"✅ file {file_name} saved successfully. ({len(required_data)} rows)")

print("\n--- finished ---")