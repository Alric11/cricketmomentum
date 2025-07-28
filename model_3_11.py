# model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Lambda, Input, Concatenate, Embedding, Flatten
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

TOTAL_BALLS = 120  # For a T20 match
SEQ_LENGTH = 10

def create_sequences(df, seq_length, ball_features, cum_features, ctx_features, chase_features):
    X_seq, X_cum, X_ctx, X_chase, y_target = [], [], [], [], []
    for i in range(len(df) - seq_length):
        seq = df.iloc[i:i+seq_length][ball_features].values
        X_seq.append(seq)
        X_cum.append(df.iloc[i+seq_length][cum_features].values)
        X_ctx.append(df.iloc[i+seq_length][ctx_features].values)
        X_chase.append(df.iloc[i+seq_length][chase_features].values)
        y_target.append(df.iloc[i+seq_length]['Target Momentum'])
    return (np.array(X_seq, dtype=np.float32),
            np.array(X_cum, dtype=np.float32),
            np.array(X_ctx, dtype=np.float32),
            np.array(X_chase, dtype=np.float32),
            np.array(y_target, dtype=np.float32))

def load_and_preprocess_data(csv_path, seq_length=SEQ_LENGTH):
    data = pd.read_csv(csv_path)
    data.sort_values(by=['Over', 'Ball'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Expected contextual numeric columns.
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in data.columns:
            data[col] = 0  # default value

    # Compute cumulative features.
    data['Cumulative Runs'] = data['Runs From Ball'].cumsum()
    data['Cumulative Wickets'] = data['Wicket'].astype(float).cumsum()
    data['Balls Bowled'] = np.arange(1, len(data) + 1)
    data['Overs Completed'] = data['Balls Bowled'] / 6.0
    data['Current Run Rate'] = data['Cumulative Runs'] / data['Overs Completed']
    
    # Compute chase-specific features.
    def compute_chase_features(row):
        if row.get('Innings', 1) == 2:
            balls_remaining = TOTAL_BALLS - row['Balls Bowled']
            overs_remaining = balls_remaining / 6.0 if balls_remaining > 0 else 0.1
            req_rr = (row['Target Score'] - row['Cumulative Runs']) / overs_remaining
            chase_diff = row['Current Run Rate'] - req_rr
            return pd.Series([1, req_rr, chase_diff])
        else:
            return pd.Series([0, 0, 0])
    data[['Is_Chasing', 'Required Run Rate', 'Chase Differential']] = data.apply(compute_chase_features, axis=1)
    
    # Define feature groups.
    ball_features = ['Batter Runs', 'Extra Runs', 'Runs From Ball', 'Balls Remaining']
    cum_features = ['Cumulative Runs', 'Cumulative Wickets', 'Current Run Rate']
    ctx_features = ['Venue_enc'] + ctx_numeric
    chase_features = ['Is_Chasing', 'Required Run Rate', 'Chase Differential']
    
    # Process categorical feature "Venue".
    if 'Venue' not in data.columns:
        data['Venue'] = "Unknown"
    le = LabelEncoder()
    data['Venue_enc'] = le.fit_transform(data['Venue'])
    
    # Scale numeric groups.
    ball_scaler = StandardScaler()
    data[ball_features] = ball_scaler.fit_transform(data[ball_features])
    
    cum_scaler = StandardScaler()
    data[cum_features] = cum_scaler.fit_transform(data[cum_features])
    
    ctx_scaler = StandardScaler()
    data[ctx_numeric] = ctx_scaler.fit_transform(data[ctx_numeric])
    
    chase_scaler = StandardScaler()
    data[chase_features] = chase_scaler.fit_transform(data[chase_features])
    
    # Compute target momentum (example formula).
    data['Target Momentum'] = 50 + (data['Current Run Rate'] - 7) * 5 - (data['Cumulative Wickets'] * 5) + (data['Chase Differential'] * 2)
    data['Target Momentum'] = data['Target Momentum'].clip(0, 100)
    
    # Drop rows with missing values.
    all_features = ball_features + cum_features + ctx_features + chase_features + ['Target Momentum']
    data = data.dropna(subset=all_features)
    
    # Unpack 5 outputs from create_sequences.
    X_seq, X_cum, X_ctx, X_chase, y_target = create_sequences(data, seq_length, ball_features, cum_features, ctx_features, chase_features)
    return X_seq, X_cum, X_ctx, X_chase, y_target, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le, ball_features, cum_features, ctx_features

def build_model(seq_length, num_ball_features, num_cum_features, num_ctx_numeric, vocab_size, ctx_embed_dim=4):
    # Branch 1: Ball-level sequence features.
    input_seq = Input(shape=(seq_length, num_ball_features), name="seq_input")
    x = LSTM(64, return_sequences=True)(input_seq)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    # Branch 2: Cumulative features.
    input_cum = Input(shape=(num_cum_features,), name="cum_input")
    y_branch = Dense(8, activation='relu')(input_cum)
    y_branch = Dense(4, activation='relu')(y_branch)
    
    # Branch 3a: Contextual categorical feature (Venue).
    input_ctx_cat = Input(shape=(1,), dtype='int32', name="ctx_cat_input")
    z_cat = Embedding(input_dim=vocab_size, output_dim=ctx_embed_dim)(input_ctx_cat)
    z_cat = Flatten()(z_cat)
    
    # Branch 3b: Contextual numeric features.
    input_ctx_num = Input(shape=(num_ctx_numeric,), name="ctx_num_input")
    z_num = Dense(8, activation='relu')(input_ctx_num)
    z_num = Dense(4, activation='relu')(z_num)
    
    ctx_combined = Concatenate()([z_cat, z_num])
    ctx_out = Dense(8, activation='relu')(ctx_combined)
    
    # Branch 4: Chase features.
    input_chase = Input(shape=(3,), name="chase_input")
    c = Dense(8, activation='relu')(input_chase)
    c = Dense(4, activation='relu')(c)
    
    # Combine all branches.
    combined = Concatenate()([x, y_branch, ctx_out, c])
    z = Dense(8, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(z)
    output = Lambda(lambda x: x * 100)(output)
    
    model = Model(inputs=[input_seq, input_cum, input_ctx_cat, input_ctx_num, input_chase], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(csv_path, seq_length=SEQ_LENGTH, epochs=20, batch_size=32):
    (X_seq, X_cum, X_ctx, X_chase, y_target, 
     ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le, 
     ball_features, cum_features, ctx_features) = load_and_preprocess_data(csv_path, seq_length)
    
    X_seq_train, X_seq_val, X_cum_train, X_cum_val, X_ctx_train, X_ctx_val, X_chase_train, X_chase_val, y_train, y_val = train_test_split(
        X_seq, X_cum, X_ctx, X_chase, y_target, test_size=0.2, random_state=42)
    
    num_ball_features = X_seq.shape[-1]
    num_cum_features = X_cum.shape[-1]
    num_ctx_numeric = len(ctx_features) - 1  # excluding encoded Venue
    vocab_size = len(le.classes_)
    
    model = build_model(seq_length, num_ball_features, num_cum_features, num_ctx_numeric, vocab_size)
    model.summary()
    
    model.fit([X_seq_train, X_cum_train, X_ctx_train[:, 0].reshape(-1, 1), X_ctx_train[:, 1:], X_chase_train],
              y_train, epochs=epochs, batch_size=batch_size,
              validation_data=([X_seq_val, X_cum_val, X_ctx_val[:, 0].reshape(-1, 1), X_ctx_val[:, 1:], X_chase_val], y_val))
    
    model.save('momentum_model.h5')
    with open('ball_scaler.pkl', 'wb') as f:
        pickle.dump(ball_scaler, f)
    with open('cum_scaler.pkl', 'wb') as f:
        pickle.dump(cum_scaler, f)
    with open('ctx_scaler.pkl', 'wb') as f:
        pickle.dump(ctx_scaler, f)
    with open('chase_scaler.pkl', 'wb') as f:
        pickle.dump(chase_scaler, f)
    with open('venue_le.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('ball_features.pkl', 'wb') as f:
        pickle.dump(ball_features, f)
    with open('cum_features.pkl', 'wb') as f:
        pickle.dump(cum_features, f)
    with open('ctx_features.pkl', 'wb') as f:
        pickle.dump(ctx_features, f)
    
    print("Training complete. Model and preprocessing objects saved.")
    return model, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le, ball_features, cum_features, ctx_features

if __name__ == "__main__":
    csv_path = 'ball_by_ball_ipl.csv'  # Update with your CSV file path
    train_model(csv_path, seq_length=10, epochs=10, batch_size=32)
