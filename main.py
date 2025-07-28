import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import openai

# =============================================================================
# GLOBAL CONSTANTS & FILE PATHS
# =============================================================================
TOTAL_BALLS = 120       # T20: 20 overs x 6 balls
SEQ_LENGTH = 10
HISTORICAL_CSV = "ball_by_ball_ipl.csv"          # Historical ball-by-ball data CSV
LIVE_CSV = "converted_match_innings2_over10.csv"  # Live match CSV (set to innings 1 data)
TEAM_CSV = "team.csv"                             # Starting XI CSV

# Default team names (will be overridden by TEAM_CSV if available)
DEFAULT_TEAM1 = "Mumbai Indians"
DEFAULT_TEAM2 = "Sunrisers Hyderabad"

# =============================================================================
# LOAD PRE-TRAINED OBJECTS & FEATURE LISTS
# =============================================================================
with open('ball_scaler.pkl', 'rb') as f:
    ball_scaler = pickle.load(f)
with open('cum_scaler.pkl', 'rb') as f:
    cum_scaler = pickle.load(f)
with open('ctx_scaler.pkl', 'rb') as f:
    ctx_scaler = pickle.load(f)
with open('chase_scaler.pkl', 'rb') as f:
    chase_scaler = pickle.load(f)
with open('venue_le.pkl', 'rb') as f:
    le = pickle.load(f)
with open('ball_features.pkl', 'rb') as f:
    ball_features = pickle.load(f)
with open('cum_features.pkl', 'rb') as f:
    cum_features = pickle.load(f)
with open('ctx_features.pkl', 'rb') as f:
    ctx_features = pickle.load(f)
chase_features = ['Is_Chasing', 'Required Run Rate', 'Chase Differential']

# =============================================================================
# OPENAI API SETUP (Direct assignment of API key)
# =============================================================================
openai.api_key = ""  # Replace with your actual API key

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def standardize_columns(df):
    df.columns = df.columns.str.strip()
    mapping = {
        "batter": "Batter",
        "bowler": "Bowler",
        "non_striker": "Non Striker",
        "runs_batter": "Batter Runs",
        "runs_extras": "Extra Runs",
        "runs_total": "Runs From Ball"
    }
    df.rename(columns=mapping, inplace=True)
    return df

def add_team_info(df):
    if "Innings" in df.columns:
        df["Innings"] = df["Innings"].astype(int)
        df["BattingTeam"] = df["Innings"].apply(lambda x: DEFAULT_TEAM1 if x == 1 else DEFAULT_TEAM2)
        df["BowlingTeam"] = df["Innings"].apply(lambda x: DEFAULT_TEAM2 if x == 1 else DEFAULT_TEAM1)
    return df

def ensure_ctx_columns(df):
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy',
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in df.columns:
            df[col] = 0
    return df

def is_valid_delivery(row):
    if "extras_wides" in row and pd.notna(row["extras_wides"]):
        try:
            return float(row["extras_wides"]) == 0
        except:
            return True
    return True

def compute_live_features(df, total_balls=TOTAL_BALLS):
    df = df.copy()
    df = standardize_columns(df)
    # If "Over" or "Ball" are missing, create them from the index
    if "Over" not in df.columns:
        df["Over"] = (df.index // 6) + 1
    if "Ball" not in df.columns:
        df["Ball"] = (df.index % 6) + 1
    df = add_team_info(df)
    df = ensure_ctx_columns(df)
    if "Innings" in df.columns:
        df.sort_values(by=["Innings", "Over", "Ball"], inplace=True)
    else:
        df.sort_values(by=["Over", "Ball"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Runs From Ball"] = pd.to_numeric(df["Runs From Ball"], errors="coerce").fillna(0)
    df["Wicket"] = pd.to_numeric(df["Wicket"], errors="coerce").fillna(0)
    df["Cumulative Runs"] = df["Runs From Ball"].cumsum()
    df["Cumulative Wickets"] = df["Wicket"].cumsum()
    df["Balls Bowled"] = np.arange(1, len(df) + 1)
    df["Overs Completed"] = df["Balls Bowled"] / 6.0
    df["Current Run Rate"] = df["Cumulative Runs"] / df["Overs Completed"]
    if "Balls Remaining" not in df.columns:
        df["Balls Remaining"] = total_balls - df["Balls Bowled"]
    for col in ["Is_Chasing", "Required Run Rate", "Chase Differential"]:
        if col not in df.columns:
            df[col] = 0
    return df

def compute_over_metrics(df):
    metrics_list = []
    for over, group in df.groupby("Over"):
        over_runs = group["Runs From Ball"].sum()
        wickets = group["Wicket"].sum()
        valid_balls = group.apply(lambda row: 1 if is_valid_delivery(row) else 0, axis=1).sum()
        over_run_rate = (over_runs / valid_balls * 6) if valid_balls > 0 else 0
        metrics_list.append({
            "Over": over,
            "Over_Runs": over_runs,
            "Valid_Balls": valid_balls,
            "Wickets": wickets,
            "Over_Run_Rate": over_run_rate
        })
    metrics_df = pd.DataFrame(metrics_list)
    if "Over" in metrics_df.columns:
        metrics_df = metrics_df.set_index("Over")
    return metrics_df

def read_starting_xi(filename):
    df = pd.read_csv(filename, header=None)
    teams = df.iloc[0].tolist()
    team1 = teams[0].strip()
    team2 = teams[1].strip()
    team1_players = []
    team2_players = []
    for i in range(1, len(df)):
        row = df.iloc[i].tolist()
        if pd.notna(row[0]):
            team1_players.append(row[0].strip())
        if pd.notna(row[1]):
            team2_players.append(row[1].strip())
    return {team1: team1_players, team2: team2_players}

def compute_historical_stats(csv_path):
    df = pd.read_csv(csv_path)
    df = standardize_columns(df)
    df = add_team_info(df)
    if "Innings" in df.columns:
        df.sort_values(by=["Innings", "Over", "Ball"], inplace=True)
    else:
        df.sort_values(by=["Over", "Ball"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    batter_stats = df.groupby(["Batter", "BattingTeam"]).agg({
        "Batter Runs": ["sum", "count"],
        "Runs From Ball": "sum"
    })
    batter_stats.columns = ["Hist_Total_Runs", "Hist_Balls_Faced", "Hist_Runs_Contributed"]
    batter_stats["Hist_Strike_Rate"] = (batter_stats["Hist_Total_Runs"] / batter_stats["Hist_Balls_Faced"] * 100).round(2)
    bowler_stats = df.groupby(["Bowler", "BowlingTeam"]).agg({
        "Runs From Ball": "sum",
        "Wicket": "sum",
        "Ball": "count"
    })
    bowler_stats.rename(columns={"Ball": "Hist_Balls_Bowled"}, inplace=True)
    bowler_stats["Hist_Overs"] = bowler_stats["Hist_Balls_Bowled"] / 6.0
    bowler_stats["Hist_Economy"] = (bowler_stats["Runs From Ball"] / bowler_stats["Hist_Overs"]).round(2)
    bowler_stats["Hist_Wickets"] = bowler_stats["Wicket"].astype(int)
    bowler_stats.drop(columns=["Wicket"], inplace=True)
    return batter_stats, bowler_stats

def predict_momentum_live(df, ball_features, cum_features, ctx_features, chase_features):
    if len(df) < SEQ_LENGTH:
        raise ValueError("Not enough live data to form a sequence.")
    X_seq = df[ball_features].values
    X_seq_scaled = ball_scaler.transform(X_seq)
    sequence = X_seq_scaled[-SEQ_LENGTH:]
    sequence = np.expand_dims(sequence, axis=0)
    current_cum = df.iloc[-1][cum_features].values.reshape(1, -1)
    current_cum_scaled = cum_scaler.transform(current_cum)
    venue_val = df.iloc[-1]["Venue"]
    venue_index = le.transform([venue_val])[0] if venue_val in le.classes_ else 0
    venue_enc = np.array([[venue_index]])
    ctx_numeric = df.iloc[-1][ctx_features[1:]].values.reshape(1, -1)
    ctx_numeric_scaled = ctx_scaler.transform(ctx_numeric)
    chase_vals = df.iloc[-1][chase_features].values.reshape(1, -1)
    chase_scaled = chase_scaler.transform(chase_vals)
    loaded_model = load_model("momentum_model.h5", custom_objects={"mse": "mse"})
    predicted = loaded_model.predict([sequence, current_cum_scaled, venue_enc, ctx_numeric_scaled, chase_scaled])
    return predicted[0][0]

def compute_recent_stats(df, player_col, stat_col, window=5):
    recent_stats = {}
    for player, group in df.groupby(player_col):
        group = group.sort_values(by=["Over", "Ball"])
        recent_stats[player] = group[stat_col].tail(window).mean()
    return recent_stats

def get_last_ball_details(df):
    if df.empty:
        return {"Last Over": "N/A", "Last Bowler": "N/A", "On-Strike Batter": "N/A", "Non-Striker": "N/A"}
    latest_over = df["Over"].max()
    latest_df = df[df["Over"] == latest_over].sort_values(by="Ball")
    if latest_df.empty:
        return {"Last Over": "N/A", "Last Bowler": "N/A", "On-Strike Batter": "N/A", "Non-Striker": "N/A"}
    last_ball = latest_df.iloc[-1]
    return {
        "Last Over": int(latest_over),
        "Last Bowler": last_ball["Bowler"],
        "On-Strike Batter": last_ball["Batter"],
        "Non-Striker": last_ball["Non Striker"]
    }

def compute_chase_info(df, team):
    team_df = df[df["BattingTeam"] == team]
    if team_df.empty or "Target Score" not in team_df.columns:
        return ""
    try:
        target = float(team_df["Target Score"].iloc[-1])
    except:
        return ""
    current_runs = float(team_df["Cumulative Runs"].iloc[-1])
    runs_remaining = target - current_runs
    balls_remaining = TOTAL_BALLS - float(team_df["Balls Bowled"].iloc[-1])
    wickets_remaining = 10 - int(team_df["Cumulative Wickets"].iloc[-1])
    return (f"Chase Info: {runs_remaining:.0f} runs to chase in {balls_remaining:.0f} balls, "
            f"{wickets_remaining} wickets remaining.")

# ----- Batter & Bowler Report Functions -----
def prepare_batter_report(batting_df, historical_batter_stats, starting_batters, user_team):
    report_lines = []
    recent_avg = compute_recent_stats(batting_df, "Batter", "Batter Runs", window=5)
    report_lines.append("=== Batter Scorecard ===")
    for batter in starting_batters:
        report_lines.append(f"Batter: {batter}")
        if batter in batting_df["Batter"].unique():
            group = batting_df[batting_df["Batter"] == batter]
            total_runs = group["Batter Runs"].sum()
            balls = group.shape[0]
            strike_rate = (total_runs / balls * 100) if balls > 0 else 0
            fours = (group["Batter Runs"] == 4).sum()
            sixes = (group["Batter Runs"] == 6).sum()
            recent = recent_avg.get(batter, None)
            report_lines.append(f"  Live: Total Runs = {total_runs}, Balls Faced = {balls}, Strike Rate = {strike_rate:.2f}")
            report_lines.append(f"  Boundaries: {fours} fours, {sixes} sixes")
            if recent is not None:
                report_lines.append(f"  Recent (last 5 balls) Avg Runs = {recent:.2f}")
            else:
                report_lines.append("  Recent performance: N/A")
        else:
            report_lines.append("  Did not bat")
        try:
            hist_info = historical_batter_stats.loc[(batter, user_team)]
            live_sr = (total_runs / balls * 100) if (batter in batting_df["Batter"].unique() and balls > 0) else 0
            diff_sr = hist_info["Hist_Strike_Rate"] - live_sr
            report_lines.append(f"  Historical (for {user_team}) - Total Runs = {hist_info['Hist_Total_Runs']}, "
                                f"Balls Faced = {hist_info['Hist_Balls_Faced']}, Strike Rate = {hist_info['Hist_Strike_Rate']}")
            if diff_sr >= 30:
                report_lines.append(f"  Insight: Strike rate is {diff_sr:.1f} points lower than historical average.")
            else:
                report_lines.append("  Insight: Batting performance is in line with historical trends.")
        except Exception:
            report_lines.append("  Historical data: Not available")
        report_lines.append("")
    return "\n".join(report_lines)

def prepare_bowler_report(bowling_df, historical_bowler_stats, starting_bowlers, opponent_team):
    report_lines = []
    live_bowlers = {}
    for bowler, group in bowling_df.groupby("Bowler"):
        runs_conceded = group["Runs From Ball"].sum()
        valid_balls = group.apply(lambda r: 1 if is_valid_delivery(r) else 0, axis=1).sum()
        overs = int(valid_balls // 6)
        balls = int(valid_balls % 6)
        overs_str = f"{overs}.{balls}"
        economy = (runs_conceded / (valid_balls / 6)) if valid_balls > 0 else 0
        wickets = int(group["Wicket"].sum())
        live_bowlers[bowler] = {"Runs": runs_conceded, "Overs": overs_str, "Economy": economy, "Wickets": wickets}
    report_lines.append("=== Bowler Scorecard ===")
    for bowler in starting_bowlers:
        report_lines.append(f"Bowler: {bowler}")
        if bowler in live_bowlers:
            info = live_bowlers[bowler]
            report_lines.append(f"  Live: Runs Conceded = {info['Runs']}, Overs (legal) = {info['Overs']}, "
                                f"Economy = {info['Economy']:.2f}, Wickets = {info['Wickets']}")
            try:
                hist_info = historical_bowler_stats.loc[(bowler, opponent_team)]
                report_lines.append(f"  Historical (against {opponent_team}) - Economy = {hist_info['Hist_Economy']}, "
                                    f"Wickets = {hist_info['Hist_Wickets']}")
            except Exception:
                report_lines.append("  Historical data: Not available")
        else:
            report_lines.append("  Did not bowl")
        report_lines.append("")
    return "\n".join(report_lines)

def predict_team_momentum(df_live, ball_features, cum_features, ctx_features, chase_features):
    try:
        mom = predict_momentum_live(df_live, ball_features, cum_features, ctx_features, chase_features)
        return mom
    except Exception:
        return None

# =============================================================================
# REPORT GENERATION FUNCTIONS
# =============================================================================
def generate_scorecard_report(live_df, historical_csv, user_team):
    df_live = live_df.copy()
    df_live = standardize_columns(df_live)
    df_live = compute_live_features(df_live, TOTAL_BALLS)
    df_live = add_team_info(df_live)
    
    starting_dict = read_starting_xi(TEAM_CSV)
    starting_batters = starting_dict.get(user_team, [])
    opponent_team = DEFAULT_TEAM2 if user_team == DEFAULT_TEAM1 else DEFAULT_TEAM1
    starting_bowlers = starting_dict.get(opponent_team, [])
    
    batting_df = df_live[df_live["BattingTeam"] == user_team]
    bowling_df = df_live[df_live["BowlingTeam"] == opponent_team]
    
    historical_batter_stats, historical_bowler_stats = compute_historical_stats(historical_csv)
    
    report_lines = []
    report_lines.append(f"=== Detailed Report for {user_team} ===\n")
    if not batting_df.empty:
        bat_mom = predict_team_momentum(batting_df, ball_features, cum_features, ctx_features, chase_features)
        report_lines.append(f"Predicted Batting Momentum for {user_team}: {bat_mom:.2f} / 100")
    else:
        report_lines.append("Predicted Batting Momentum: N/A")
    if not bowling_df.empty:
        bowl_mom = predict_team_momentum(bowling_df, ball_features, cum_features, ctx_features, chase_features)
        report_lines.append(f"Predicted Bowling Momentum for {opponent_team} (i.e. {user_team}'s fielding): {100 - bowl_mom:.2f} / 100")
    else:
        report_lines.append("Predicted Bowling Momentum: N/A")
    report_lines.append("")
    report_lines.append("=== Batting Scorecard ===")
    report_lines.append(prepare_batter_report(batting_df, historical_batter_stats, starting_batters, user_team))
    report_lines.append("")
    report_lines.append("=== Bowling Scorecard (Opposing Team) ===")
    bowler_rep = prepare_bowler_report(bowling_df, historical_bowler_stats, starting_bowlers, opponent_team)
    if bowler_rep.strip() == "":
        report_lines.append("No bowling data available for team: " + opponent_team)
    else:
        report_lines.append(bowler_rep)
    report_lines.append("")
    over_metrics = compute_over_metrics(df_live)
    report_lines.append("=== Over-by-Over Metrics ===")
    for over, row in over_metrics.iterrows():
        if over < TOTAL_BALLS / 6:
            report_lines.append(f"Over {int(over)+1}: Runs = {row['Over_Runs']}, Valid Balls = {int(row['Valid_Balls'])}, "
                                f"Wickets = {int(row['Wickets'])}, Over Run Rate = {row['Over_Run_Rate']:.2f}")
    report_lines.append("")
    last_details = get_last_ball_details(df_live)
    report_lines.append("=== Last Ball Details ===")
    report_lines.append(f"Last Over: {last_details['Last Over']}")
    report_lines.append(f"Last Bowler: {last_details['Last Bowler']}")
    report_lines.append(f"On-Strike Batter: {last_details['On-Strike Batter']}")
    report_lines.append(f"Non-Striker: {last_details['Non-Striker']}")
    report_lines.append("")
    # Check if chase info is available without causing an index error.
    if "Target Score" in df_live.columns and not df_live.empty and pd.notna(df_live.iloc[-1].get("Target Score", np.nan)):
        chase_text = compute_chase_info(df_live, user_team)
        if chase_text:
            report_lines.append("=== Chase Information ===")
            report_lines.append(chase_text)
            report_lines.append("")
    return "\n".join(report_lines)

def generate_full_match_report(live_csv, historical_csv, team1, team2):
    df_live = pd.read_csv(live_csv)
    df_live = standardize_columns(df_live)
    df_live = compute_live_features(df_live, TOTAL_BALLS)
    df_live = add_team_info(df_live)
    if "Innings" in df_live.columns:
        df_innings1 = df_live[df_live["Innings"] == 1].copy()
        df_innings2 = df_live[df_live["Innings"] == 2].copy()
    else:
        df_innings1 = df_live.copy()
        df_innings2 = pd.DataFrame()
    
    report_team1 = generate_scorecard_report(df_innings1, historical_csv, team1)
    report_team2 = generate_scorecard_report(df_innings2, historical_csv, team2)
    
    if not df_innings1.empty:
        mom_team1 = predict_team_momentum(df_innings1[df_innings1["BattingTeam"] == team1],
                                          ball_features, cum_features, ctx_features, chase_features)
        mom_text_team1 = f"Predicted Momentum for {team1} (batting, innings 1): {mom_team1:.2f} / 100"
    else:
        mom_text_team1 = f"Predicted Momentum for {team1}: N/A"
    
    if not df_innings1.empty:
        mom_team2 = predict_team_momentum(df_innings1[df_innings1["BowlingTeam"] == team2],
                                          ball_features, cum_features, ctx_features, chase_features)
        mom_text_team2 = f"Predicted Momentum for {team2} (bowling, innings 1): {100 - mom_team2:.2f} / 100"
    else:
        mom_text_team2 = f"Predicted Momentum for {team2}: N/A"
    
    full_report = (f"=== FULL MATCH REPORT ===\n\n"
                   f"--- Inning 1 ---\n\n"
                   f"{report_team1}\n\n"
                   f"Overall Momentum Scores (Innings 1):\n"
                   f"{mom_text_team1}\n{mom_text_team2}\n")
    
    if not df_innings2.empty:
        report_team2_bat = generate_scorecard_report(df_innings2, historical_csv, team2)
        if not df_innings2.empty:
            mom_team2_bat = predict_team_momentum(df_innings2[df_innings2["BattingTeam"] == team2],
                                                  ball_features, cum_features, ctx_features, chase_features)
            mom_text_team2_bat = f"Predicted Momentum for {team2} (batting, innings 2): {mom_team2_bat:.2f} / 100"
        else:
            mom_text_team2_bat = f"Predicted Momentum for {team2}: N/A"
        if not df_innings2.empty:
            mom_team1_bowl = predict_team_momentum(df_innings2[df_innings2["BowlingTeam"] == team1],
                                                   ball_features, cum_features, ctx_features, chase_features)
            mom_text_team1_bowl = f"Predicted Momentum for {team1} (bowling, innings 2): {100 - mom_team1_bowl:.2f} / 100"
        else:
            mom_text_team1_bowl = f"Predicted Momentum for {team1}: N/A"
        full_report += (f"\n--- Inning 2 ---\n\n"
                        f"{report_team2_bat}\n\n"
                        f"Overall Momentum Scores (Innings 2):\n"
                        f"{mom_text_team2_bat}\n{mom_text_team1_bowl}\n")
    
    return full_report

# =============================================================================
# GPT RECOMMENDATION FUNCTION
# =============================================================================
def generate_gpt_recommendations(match_report, batting_team, bowling_team):
    prompt = (
        f"Assume you are a cricket analytics coach. Based on the following match report, "
        f"provide 5 actionable recommendations and tactics for the batting team and 5 for the bowling team. "
        f"Focus only on batting advice when discussing the batting team and only on bowling advice for the fielding team. "
        f"Format your response as a numbered list, where each recommendation includes a main topic, actions, and purpose.\n\n"
        f"Team XI:\n"
        f"Batting Team: {batting_team}\n"
        f"Bowling Team: {bowling_team}\n\n"
        f"Match Report:\n{match_report}\n\n"
        f"Provide recommendations based solely on the latest data in the report (e.g. last ball details, momentum scores, current player performance)."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert cricket analytics coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message["content"]

# =============================================================================
# MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    # Read live CSV and ensure team info.
    df_live = pd.read_csv(LIVE_CSV)
    df_live = standardize_columns(df_live)
    if "Innings" in df_live.columns:
        df_live["BattingTeam"] = df_live["Innings"].apply(lambda x: DEFAULT_TEAM1 if int(x) == 1 else DEFAULT_TEAM2)
        df_live["BowlingTeam"] = df_live["Innings"].apply(lambda x: DEFAULT_TEAM2 if int(x) == 1 else DEFAULT_TEAM1)
    else:
        df_live["BattingTeam"] = DEFAULT_TEAM1
        df_live["BowlingTeam"] = DEFAULT_TEAM2
    df_live.to_csv("live_temp.csv", index=False)
    
    full_report = generate_full_match_report("live_temp.csv", HISTORICAL_CSV, DEFAULT_TEAM1, DEFAULT_TEAM2)
    print("\n================ FULL MATCH REPORT ================\n")
    print(full_report)
    
    # Determine current innings based on the last row of the live DataFrame.
    last_innings = int(df_live["Innings"].iloc[-1]) if "Innings" in df_live.columns and len(df_live) > 0 else 1
    if last_innings == 1:
        batting_team = DEFAULT_TEAM1
        bowling_team = DEFAULT_TEAM2
    else:
        batting_team = DEFAULT_TEAM2
        bowling_team = DEFAULT_TEAM1
    
    gpt_recs = generate_gpt_recommendations(full_report, batting_team, bowling_team)
    print("\n================ GPT RECOMMENDATIONS ================\n")
    print(gpt_recs)
