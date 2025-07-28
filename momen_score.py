import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load dataset
df = pd.read_csv('ball_by_ball_ipl.csv')

# Display basic information about the dataset
print(df.info())

# Fill missing values
df.fillna({'Extra Type': 'None', 'Wicket': 0, 'Method': 'None', 'Player Out': 'None'}, inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Venue', 'Bat First', 'Bat Second', 'Batter', 'Non Striker', 'Bowler', 'Extra Type', 'Method', 'Player Out', 'Winner']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Match ID, Innings, Over, Ball to maintain sequence order
df.sort_values(by=['Match ID', 'Innings', 'Over', 'Ball'], inplace=True)

# Calculate cumulative runs and wickets for each innings
df['Cumulative Runs'] = df.groupby(['Match ID', 'Innings'])['Batter Runs'].cumsum()
df['Cumulative Wickets'] = df.groupby(['Match ID', 'Innings'])['Wicket'].cumsum()

# Calculate run rate
df['Balls Bowled'] = df.groupby(['Match ID', 'Innings']).cumcount() + 1
df['Run Rate'] = df['Cumulative Runs'] / (df['Balls Bowled'] / 6)

# Calculate required run rate
df['Remaining Runs'] = df['Target Score'] - df['Cumulative Runs']
df['Remaining Overs'] = df['Balls Remaining'] / 6
df['Required Run Rate'] = df['Remaining Runs'] / df['Remaining Overs']

# Fill infinite values with zero (occurs when Remaining Overs is zero)
df.replace([np.inf, -np.inf], 0, inplace=True)

def calculate_momentum_score(row):
    # Example: Positive score for runs, negative for wickets, adjusted by run rate difference
    run_rate_diff = row['Run Rate'] - row['Required Run Rate']
    momentum = (row['Batter Runs'] - (row['Wicket'] * 5)) + run_rate_diff
    return momentum

df['Momentum Score'] = df.apply(calculate_momentum_score, axis=1)

from sklearn.model_selection import train_test_split

# Select features for the model
features = ['Over', 'Ball', 'Batter', 'Non Striker', 'Bowler', 'Batter Runs', 'Extra Runs', 'Wicket',
            'Cumulative Runs', 'Cumulative Wickets', 'Run Rate', 'Required Run Rate', 'Balls Remaining']

# Standardize numerical features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Prepare input (X) and output (y) variables
X = df[features].values
y = df['Momentum Score'].values

# Reshape X to 3D array for LSTM: [samples, time steps, features]
# Assuming each match is a separate sequence
X_sequences = []
y_sequences = []
sequence_length = 30  # Number of timesteps in each sequence

for match_id in df['Match ID'].unique():
    match_data = df[df['Match ID'] == match_id]
    for i in range(len(match_data) - sequence_length):
        X_sequences.append(match_data[features].iloc[i:i+sequence_length].values)
        y_sequences.append(match_data['Momentum Score'].iloc[i+sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Predict momentum scores for the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')
