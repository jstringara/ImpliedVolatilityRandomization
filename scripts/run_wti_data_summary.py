import os
from datetime import date
import pandas as pd
import numpy as np

"""
This script extracts information from WTI option data files and creates a summary table
showing expiry dates, days to maturity, number of quotes, and strike price ranges.
"""

if __name__ == "__main__":
    # Set the reference date (matching the one in run_WTI_calibration.py)
    today = date(2025, 9, 23)
    
    # Define the expiry dates and filenames (as used in run_WTI_calibration.py)
    expiry_dates = [
        date(2025, 11, 17),
        date(2025, 12, 16),
        date(2026, 1, 14),
        date(2026, 2, 17),
    ]
    filenames = [
        "WTI_20251117.csv",
        "WTI_20251216.csv",
        "WTI_20260114.csv",
        "WTI_20260217.csv",
    ]
    months = ["November", "December", "January", "February"]
    
    # Initialize lists to store data for the table
    expiry_list = []
    days_to_maturity_list = []
    quotes_list = []
    min_strike_list = []
    max_strike_list = []
    
    # Process each file
    for edate, filename, month in zip(expiry_dates, filenames, months):
        # Calculate days to maturity
        days_to_maturity = (edate - today).days
        
        try:
            # Load the data
            data_path = os.path.join("data", "WTI", filename)
            data = pd.read_csv(data_path, index_col=0).squeeze()
            
            # Extract information
            strikes = np.array(data.index)
            num_quotes = len(strikes)
            min_strike = strikes.min()
            max_strike = strikes.max()
            
            # Store the information
            expiry_list.append(f"{month} {edate.day}, {edate.year}")
            days_to_maturity_list.append(days_to_maturity)
            quotes_list.append(num_quotes)
            min_strike_list.append(min_strike)
            max_strike_list.append(max_strike)
            
            print(f"Processed {month} expiry: {num_quotes} quotes from {min_strike:.2f} to {max_strike:.2f}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create a DataFrame for the table
    table_df = pd.DataFrame({
        'Expiry Date': expiry_list,
        'Days to Maturity': days_to_maturity_list,
        'Quotes': quotes_list,
        'Min Strike': min_strike_list,
        'Max Strike': max_strike_list
    })
    
    # Print the formatted table
    print("\nWTI Option Data Summary:")
    print(table_df.to_string(index=False))
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[h!]\n\\centering\n"
    latex_table += "\\begin{tabular}{lccccc}\n\\toprule\n"
    latex_table += "Expiry Date & Days to Maturity & Quotes & Min Strike & Max Strike \\\\\n"
    latex_table += "\\midrule\n"
    
    for i in range(len(expiry_list)):
        latex_table += f"{expiry_list[i]} & {days_to_maturity_list[i]} & {quotes_list[i]} & "
        latex_table += f"{min_strike_list[i]:.2f} & {max_strike_list[i]:.2f} \\\\\n"
    
    latex_table += "\\bottomrule\n\\end{tabular}\n"
    latex_table += "\\caption{Summary of WTI option data used for calibration}\n"
    latex_table += "\\label{tab:wti_market_data}\n\\end{table}"
    
    # Save the LaTeX table to a file
    output_table_dir = "outputs/tables"
    os.makedirs(output_table_dir, exist_ok=True)
    output_file = os.path.join(output_table_dir, "WTI_market_data.tex")
    
    with open(output_file, "w") as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to {output_file}")
