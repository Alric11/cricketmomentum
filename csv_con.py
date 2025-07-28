import json
import pandas as pd
import numpy as np

TOTAL_BALLS = 120  # Fixed total balls per innings (e.g. T20 match)

def flatten_innings(innings, inning_no):
    """
    Flattens an innings (from the JSON) into a list of dictionaries (one per delivery).
    For each delivery, it computes cumulative runs and wickets,
    assigns Over (1-indexed) and Ball numbers, and computes Balls Remaining.
    """
    rows = []
    cumulative_runs = 0
    cumulative_wickets = 0
    for over_data in innings.get("overs", []):
        over_no = over_data.get("over", 0) + 1  # Convert from 0-indexed to 1-indexed
        deliveries = over_data.get("deliveries", [])
        for ball_idx, delivery in enumerate(deliveries, start=1):
            runs_total = delivery.get("runs", {}).get("total", 0)
            cumulative_runs += runs_total
            wicket_flag = 1 if "wickets" in delivery and delivery["wickets"] else 0
            cumulative_wickets += wicket_flag

            row = {
                "Match ID": "",            # Not available in JSON; leave blank or set if available
                "Date": "",                # Not available in JSON
                "Venue": "",               # Not provided at delivery level
                "Bat First": "",           # Not provided at delivery level
                "Bat Second": "",          # Not provided at delivery level
                "Innings": inning_no,
                "Over": over_no,
                "Ball": ball_idx,
                "Batter": delivery.get("batter", ""),
                "Non Striker": delivery.get("non_striker", ""),
                "Bowler": delivery.get("bowler", ""),
                "Batter Runs": delivery.get("runs", {}).get("batter", 0),
                "Extra Runs": delivery.get("runs", {}).get("extras", 0),
                "Runs From Ball": runs_total,
                "Ball Rebowled": "",        # Not provided
                "Extra Type": "",            # Not provided
                "Wicket": wicket_flag,
                "Method": delivery.get("wickets", [{}])[0].get("kind", "") if wicket_flag else "",
                "Player Out": delivery.get("wickets", [{}])[0].get("player_out", "") if wicket_flag else "",
                "Innings Runs": cumulative_runs,
                "Innings Wickets": cumulative_wickets,
                "Target Score": "",         # Not provided
                "Runs to Get": "",          # Not provided
                "Balls Remaining": TOTAL_BALLS - (((over_no - 1) * 6) + ball_idx),
                "Winner": "",               # Not provided at delivery level
                "Chased Successfully": "",  # Not provided
                "Total Batter Runs": "",    # Aggregated stats not provided here
                "Total Non Striker Runs": "",  # Aggregated stats not provided here
                "Batter Balls Faced": "",      # Aggregated stats not provided here
                "Non Striker Balls Faced": "", # Aggregated stats not provided here
                "Player Out Runs": "",         # Aggregated stats not provided here
                "Player Out Balls Faced": "",  # Aggregated stats not provided here
                "Bowler Runs Conceded": "",    # Aggregated stats not provided here
                "Valid Ball": ""               # Not provided; could be used for legal delivery flag
            }
            rows.append(row)
    return rows

def json_to_csv_for_overs(json_path, overs_list=[5, 10, 15]):
    """
    Converts a JSON file (with innings data) into CSV files for various over thresholds.
    For innings 1, each output file includes only deliveries with Over <= threshold.
    For innings 2, each output file includes all innings 1 deliveries plus innings 2 deliveries 
    with Over <= threshold.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    innings = data.get("innings", [])
    if not innings:
        raise ValueError("No innings data found in JSON.")
    
    # Flatten innings 1
    rows_innings1 = flatten_innings(innings[0], inning_no=1)
    df_innings1 = pd.DataFrame(rows_innings1)
    
    # Flatten innings 2 (if available)
    if len(innings) > 1:
        rows_innings2 = flatten_innings(innings[1], inning_no=2)
        df_innings2 = pd.DataFrame(rows_innings2)
    else:
        df_innings2 = pd.DataFrame()  # No second innings

    # Process innings 1 CSV outputs
    for threshold in overs_list:
        df_subset = df_innings1[df_innings1["Over"] <= threshold].copy()
        df_subset.sort_values(by=["Over", "Ball"], inplace=True)
        out_filename = f"converted_match_innings1_over{threshold}.csv"
        df_subset.to_csv(out_filename, index=False)
        print(f"Saved {out_filename}")

    # Process innings 2 CSV outputs (if innings2 exists)
    if not df_innings2.empty:
        for threshold in overs_list:
            # For inning 2, include all inning 1 data and inning 2 deliveries with Over <= threshold.
            df_innings2_subset = df_innings2[df_innings2["Over"] <= threshold].copy()
            combined_df = pd.concat([df_innings1, df_innings2_subset], ignore_index=True)
            combined_df.sort_values(by=["Innings", "Over", "Ball"], inplace=True)
            out_filename = f"converted_match_innings2_over{threshold}.csv"
            combined_df.to_csv(out_filename, index=False)
            print(f"Saved {out_filename}")

if __name__ == "__main__":
    json_input = "1254088.json"  # Replace with your JSON file path
    json_to_csv_for_overs(json_input, overs_list=[5, 10, 15])