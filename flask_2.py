import os
import json
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import openai
from flask import Flask, jsonify

# =============================================================================
# GLOBAL CONSTANTS & FILE PATHS
# =============================================================================
TOTAL_BALLS = 120       # T20: 20 overs x 6 balls
SEQ_LENGTH = 10
HISTORICAL_CSV = "ball_by_ball_ipl.csv"          # Historical ball-by-ball data CSV
LIVE_CSV = "converted_match_innings2_over10.csv"  # Live match CSV (set to innings 1 data)
TEAM_CSV = "team.csv"                             # Starting XI CSV

# Default team names (can be overridden by TEAM_CSV)
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
# CUSTOM JSON ENCODER (to handle NumPy types)
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
        return {"Last Over": None, "Last Bowler": None, "On-Striker": None, "Non-Striker": None}
    latest_over = df["Over"].max()
    latest_df = df[df["Over"] == latest_over].sort_values(by="Ball")
    if latest_df.empty:
        return {"Last Over": None, "Last Bowler": None, "On-Striker": None, "Non-Striker": None}
    last_ball = latest_df.iloc[-1]
    return {
        "Last Over": int(latest_over),
        "Last Bowler": last_ball["Bowler"],
        "On-Striker": last_ball["Batter"],
        "Non-Striker": last_ball["Non Striker"]
    }

def compute_chase_info(df, team):
    team_df = df[df["BattingTeam"] == team]
    if team_df.empty or "Target Score" not in team_df.columns:
        return {}
    try:
        target = float(team_df["Target Score"].iloc[-1])
    except:
        return {}
    current_runs = float(team_df["Cumulative Runs"].iloc[-1])
    runs_remaining = target - current_runs
    balls_remaining = TOTAL_BALLS - float(team_df["Balls Bowled"].iloc[-1])
    wickets_remaining = 10 - int(team_df["Cumulative Wickets"].iloc[-1])
    return {
        "runs_to_chase": runs_remaining,
        "balls_remaining": balls_remaining,
        "wickets_remaining": wickets_remaining
    }

# ----- Structured Report Functions -----
def prepare_batters_analysis(batting_df, historical_batter_stats, starting_batters, user_team):
    batters_list = []
    recent_avg = compute_recent_stats(batting_df, "Batter", "Batter Runs", window=5)
    for batter in starting_batters:
        batter_dict = {"name": batter}
        if batter in batting_df["Batter"].unique():
            group = batting_df[batting_df["Batter"] == batter]
            total_runs = group["Batter Runs"].sum()
            balls = group.shape[0]
            strike_rate = (total_runs / balls * 100) if balls > 0 else 0
            fours = int((group["Batter Runs"] == 4).sum())
            sixes = int((group["Batter Runs"] == 6).sum())
            batter_dict["live_stats"] = {
                "balls_faced": balls,
                "runs": int(total_runs),
                "strike_rate": round(strike_rate, 2),
                "boundaries": f"{fours} fours, {sixes} sixes"
            }
            try:
                hist_info = historical_batter_stats.loc[(batter, user_team)]
                live_sr = (total_runs / balls * 100) if balls > 0 else 0
                diff_sr = hist_info["Hist_Strike_Rate"] - live_sr
                if live_sr >= hist_info["Hist_Strike_Rate"]:
                    batter_dict["insight"] = "Performing at or above historical norms."
                else:
                    batter_dict["insight"] = f"Strike rate is {diff_sr:.1f} points lower than historical average."
            except Exception:
                batter_dict["insight"] = "Historical data not available."
        else:
            batter_dict["live_stats"] = {"balls_faced": 0, "runs": 0, "strike_rate": 0, "boundaries": "Did not bat"}
            batter_dict["insight"] = "Did not bat."
        try:
            hist_info = historical_batter_stats.loc[(batter, user_team)]
            batter_dict["historical_stats"] = {
                "runs": int(hist_info["Hist_Total_Runs"]),
                "strike_rate": float(hist_info["Hist_Strike_Rate"])
            }
        except Exception:
            batter_dict["historical_stats"] = {}
        batters_list.append(batter_dict)
    return batters_list

def prepare_bowlers_analysis(bowling_df, historical_bowler_stats, starting_bowlers, opponent_team):
    bowlers_list = []
    live_bowlers = {}
    for bowler, group in bowling_df.groupby("Bowler"):
        runs_conceded = group["Runs From Ball"].sum()
        valid_balls = group.apply(lambda r: 1 if is_valid_delivery(r) else 0, axis=1).sum()
        overs = int(valid_balls // 6)
        balls = int(valid_balls % 6)
        overs_val = overs + balls / 6.0
        economy = (runs_conceded / overs_val) if overs_val > 0 else 0
        wickets = int(group["Wicket"].sum())
        live_bowlers[bowler] = {
            "runs_conceded": int(runs_conceded),
            "overs": round(overs_val, 2),
            "economy": round(economy, 2),
            "wickets": wickets
        }
    for bowler in starting_bowlers:
        bowler_dict = {"name": bowler}
        if bowler in live_bowlers:
            bowler_dict["live_stats"] = live_bowlers[bowler]
            try:
                hist_info = historical_bowler_stats.loc[(bowler, opponent_team)]
                bowler_dict["historical_stats"] = {
                    "economy": float(hist_info["Hist_Economy"]),
                    "wickets": int(hist_info["Hist_Wickets"])
                }
                if live_bowlers[bowler]["economy"] <= hist_info["Hist_Economy"]:
                    bowler_dict["insight"] = "Live economy is lower than historical average."
                else:
                    bowler_dict["insight"] = "Bowling economy is higher than historical average."
            except Exception:
                bowler_dict["historical_stats"] = {}
                bowler_dict["insight"] = "Historical data not available."
        else:
            bowler_dict["live_stats"] = {"runs_conceded": 0, "overs": 0, "economy": 0, "wickets": 0}
            bowler_dict["insight"] = "Did not bowl."
        bowlers_list.append(bowler_dict)
    return bowlers_list

def generate_match_summary(df):
    if df.empty:
        return {}
    total_runs = df["Runs From Ball"].sum()
    total_wickets = int(df["Wicket"].sum())
    total_balls = int(df["Balls Bowled"].iloc[-1])
    current_run_rate = df["Current Run Rate"].iloc[-1]
    overs_completed = df["Overs Completed"].iloc[-1]
    return {
        "total_runs": int(total_runs),
        "total_wickets": total_wickets,
        "total_balls": total_balls,
        "overs": round(overs_completed, 2),
        "current_run_rate": round(current_run_rate, 2)
    }

def generate_structured_report(inning_df, historical_csv, batting_team, bowling_team):
    hist_bat, hist_bowl = compute_historical_stats(historical_csv)
    report = {
        "batters_analysis": prepare_batters_analysis(
            inning_df[inning_df["BattingTeam"] == batting_team],
            hist_bat,
            starting_batters=read_starting_xi(TEAM_CSV).get(batting_team, []),
            user_team=batting_team
        ),
        "bowlers_analysis": prepare_bowlers_analysis(
            inning_df[inning_df["BowlingTeam"] == bowling_team],
            hist_bowl,
            starting_bowlers=read_starting_xi(TEAM_CSV).get(bowling_team, []),
            opponent_team=bowling_team
        ),
        "match_summary": generate_match_summary(inning_df),
        "momentum": predict_momentum_live(inning_df, ball_features, cum_features, ctx_features, chase_features) if not inning_df.empty else None,
        "last_ball_details": get_last_ball_details(inning_df),
        "chase_info": compute_chase_info(inning_df, batting_team)
    }
    return report

def generate_gpt_recommendations(match_summary_text, batting_team, bowling_team):
    prompt = (
        f"Assume you are a cricket analytics coach. Based on the following match summary, "
        f"provide 5 actionable recommendations for the batting team and 5 for the bowling team. "
        f"Focus only on batting advice for the batting team and only on bowling advice for the fielding team. "
        f"Format your response as two numbered lists (one for batting and one for bowling), where each recommendation includes a main topic, actions, and purpose.\n\n"
        f"Team XI:\nBatting Team: {batting_team}\nBowling Team: {bowling_team}\n\n"
        f"Match Summary:\n{match_summary_text}\n\n"
        f"Provide recommendations based solely on the latest data."
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

# ----- Final JSON Generation -----
def generate_final_json():
    df_live = pd.read_csv(LIVE_CSV)
    df_live = standardize_columns(df_live)
    if "Innings" in df_live.columns:
        df_live["BattingTeam"] = df_live["Innings"].apply(lambda x: DEFAULT_TEAM1 if int(x)==1 else DEFAULT_TEAM2)
        df_live["BowlingTeam"] = df_live["Innings"].apply(lambda x: DEFAULT_TEAM2 if int(x)==1 else DEFAULT_TEAM1)
    else:
        df_live["BattingTeam"] = DEFAULT_TEAM1
        df_live["BowlingTeam"] = DEFAULT_TEAM2
    df_live = compute_live_features(df_live, TOTAL_BALLS)
    df_live = add_team_info(df_live)

    if "Innings" in df_live.columns:
        df_innings1 = df_live[df_live["Innings"] == 1].copy()
        df_innings2 = df_live[df_live["Innings"] == 2].copy()
    else:
        df_innings1 = df_live.copy()
        df_innings2 = pd.DataFrame()

    structured_inning1 = generate_structured_report(df_innings1, HISTORICAL_CSV, DEFAULT_TEAM1, DEFAULT_TEAM2)
    structured_inning2 = generate_structured_report(df_innings2, HISTORICAL_CSV, DEFAULT_TEAM2, DEFAULT_TEAM1) if not df_innings2.empty else {}

    match_summary_inn1 = json.dumps(generate_match_summary(df_innings1), cls=NumpyEncoder)
    last_ball_inn1 = json.dumps(get_last_ball_details(df_innings1), cls=NumpyEncoder)
    text_summary_inn1 = f"Match Summary: {match_summary_inn1}\nLast Ball Details: {last_ball_inn1}\n"
    if not df_innings2.empty:
        match_summary_inn2 = json.dumps(generate_match_summary(df_innings2), cls=NumpyEncoder)
        last_ball_inn2 = json.dumps(get_last_ball_details(df_innings2), cls=NumpyEncoder)
        text_summary_inn2 = f"Match Summary: {match_summary_inn2}\nLast Ball Details: {last_ball_inn2}\n"
    else:
        text_summary_inn2 = ""

    gpt_recs_inn1 = generate_gpt_recommendations(text_summary_inn1, DEFAULT_TEAM1, DEFAULT_TEAM2)
    gpt_recs_inn2 = generate_gpt_recommendations(text_summary_inn2, DEFAULT_TEAM2, DEFAULT_TEAM1) if text_summary_inn2 else ""

    momentum_inn1 = structured_inning1.get("momentum", None)
    momentum_inn2 = structured_inning2.get("momentum", None)
    if momentum_inn1 is not None and momentum_inn2 is not None:
        overall_momentum = (momentum_inn1 + momentum_inn2) / 2
    else:
        overall_momentum = momentum_inn1 if momentum_inn1 is not None else momentum_inn2

    final_result = {
        "inning1": {
            "structured_report": structured_inning1,
            "gpt_recommendations": gpt_recs_inn1
        },
        "inning2": {
            "structured_report": structured_inning2,
            "gpt_recommendations": gpt_recs_inn2
        },
        "overall_momentum": overall_momentum
    }
    # Convert to JSON string using the custom encoder and then load back as a dict.
    final_json_str = json.dumps(final_result, cls=NumpyEncoder)
    final_json = json.loads(final_json_str)
    return final_json

# =============================================================================
# FLASK API SETUP
# =============================================================================
app = Flask(__name__)

@app.route('/report', methods=['GET'])
def report():
    try:
        result = generate_final_json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)
