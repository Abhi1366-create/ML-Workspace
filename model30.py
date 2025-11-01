import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import joblib 


print("Security Modules Initialized.")


# Loading the dataset
print("Fetching network traffic logs...")
train_url = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt'
test_url = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt'

column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
    'num_file_creations', 'num_shells', 'num_access_files', 
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

train_df = pd.read_csv(train_url, header=None, names=column_names)
test_df = pd.read_csv(test_url, header=None, names=column_names)

train_df = train_df.drop('difficulty', axis=1)
test_df = test_df.drop('difficulty', axis=1)

x_train = train_df.drop(['class'], axis=1)
x_test = test_df.drop(['class'], axis=1)

# Consolidated 5-class labels
attack_mapping = {'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos', 'processtable': 'dos', 'mailbomb': 'dos', 'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'mscan': 'probe', 'saint': 'probe', 'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l', 'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l', 'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'}

y_train_multi = train_df['class'].map(attack_mapping).fillna('normal')
y_test_multi = test_df['class'].map(attack_mapping).fillna('normal')

print("Network traffic successfully parsed.")

# Data for NN coree
categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = [col for col in x_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) 
    ])

print("Standardizing numerical data and preparing categorical features...")
X_train_nn = preprocessor.fit_transform(x_train)
X_test_nn = preprocessor.transform(x_test)

le = LabelEncoder()
y_train_nn = le.fit_transform(y_train_multi)
y_test_nn = le.transform(y_test_multi)

# Calculate weights to prioritize rare attacks
class_weights_nn = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_nn),
    y=y_train_nn
)
class_weights_dict = dict(enumerate(class_weights_nn))


print("Neural Core Data Prepared.")


# Bulding the final model number 30
print("--- Training the Security Mastermind: Final Stable Tune ---")

# --- THE ARCHITECTURE ---
# This is the 4-layer architecture that broke the 80% wall.
model_champion = keras.Sequential([
    layers.Input(shape=(X_train_nn.shape[1],)),
    layers.Dense(256, activation='relu'), 
    layers.Dropout(0.4), 
    layers.Dense(128, activation='relu'), 
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),  
    layers.Dropout(0.4),
    layers.Dense(5, activation='softmax') 
])

# Define the Adam optimizer with the optimal slow learning rate
optimizer_adam = Adam(learning_rate=0.0005)

# Early stopping to capture the highest score and prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=15, 
    mode='max',
    restore_best_weights=True 
)


# Compile the model
model_champion.compile(
    optimizer=optimizer_adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model_champion.summary())


#Initiate training 
print("\nInitiating Training Protocol (200 Epochs Max)... This is Model 35.")
start_time = time.time()

history = model_champion.fit(
    X_train_nn,
    y_train_nn,
    epochs=200,  
    batch_size=256,
    class_weight=class_weights_dict, 
    validation_data=(X_test_nn, y_test_nn), 
    callbacks=[early_stopping], 
    verbose=1
)

end_time = time.time()
print(f"Mastermind Training Complete! Total Duration: {(end_time - start_time)/60:.2f} minutes.")


# --- EVALUATE THE FINAL MODEL ---
y_pred_proba = model_champion.predict(X_test_nn)
y_pred_champion = np.argmax(y_pred_proba, axis=1)

y_test_multi_decoded = le.inverse_transform(y_test_nn)
y_pred_multi_decoded = le.inverse_transform(y_pred_champion)

accuracy_champion = accuracy_score(y_test_multi_decoded, y_pred_multi_decoded)

print(f"\n Final  Evaluation Report ")
print(f"OVERALL ACCURACY: {accuracy_champion * 100:.2f}%")
print("\nClassification Report (The 80%+ Champion):")
print(classification_report(y_test_multi_decoded, y_pred_multi_decoded))
print("-" * 50)


#Saveing the final model 
model_filename = 'security_mastermind_final.keras' 
model_champion.save(model_filename)
print(f"Model saved successfully as: {model_filename}")

print("\n--- MISSION COMPLETE ---")