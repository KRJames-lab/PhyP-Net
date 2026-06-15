import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import joblib
import os

class DataNormalizer:
    def __init__(self, feature_range=(0, 1)):
        self.positive_scaler = MinMaxScaler(feature_range=feature_range)    # 0, 1
        self.posi_nega_scaler = MaxAbsScaler()  # 2-13 (Output feature gathered)
        
        self.power_scaler = MaxAbsScaler()      # index : -3

    def fit(self, data_list):
        """
        Fit scaler with data_list
        """
        # data[:, 1:] -> Delete time
        all_features = [seq[:, 1:] for seq in data_list]
        concatenated_data = np.vstack(all_features)

        # Positive features scaler (Speed^2, Pedal)
        self.positive_scaler.fit(concatenated_data[:, :2])

        # Negative - Positive scaler (Speed, Acceleration, Jerk, Gradient, Delta_Pedal, Torque, Axle(RPM), Power, Delta_Power, Power, Torque, Axle(RPM))
        self.posi_nega_scaler.fit(concatenated_data[:, 2:])

        # Power scaler
        self.power_scaler.fit(concatenated_data[:, -3].reshape(-1, 1))

        print("[DataNormalizer] Scaler fitted successfully.")

    def transform(self, data_list):
        normalized_list = []
        for seq in data_list:
            features = seq[:, 1:].copy()
            
            norm_seq = np.zeros_like(features)
            
            norm_seq[:, :2] = self.positive_scaler.transform(features[:, :2])
            norm_seq[:, 2:] = self.posi_nega_scaler.transform(features[:, 2:])
            norm_seq[:, -3] = self.power_scaler.transform(features[:, -3].reshape(-1, 1)).flatten()
            
            normalized_list.append(norm_seq)
            
        return np.array(normalized_list, dtype=object)

    def save_scaler(self, positive_path="positive_scaler.pkl", posi_nega_path="posi_nega_scaler.pkl", target_power_path="power_scaler.pkl"):
        joblib.dump(self.positive_scaler, positive_path)
        joblib.dump(self.posi_nega_scaler, posi_nega_path)
        joblib.dump(self.power_scaler, target_power_path)
        print(f"Scaler saved to {positive_path} and {posi_nega_path} and {target_power_path}")

    def load_scaler(self, positive_path="positive_scaler.pkl", posi_nega_path="posi_nega_scaler.pkl", target_power_path="power_scaler.pkl"):
        if not os.path.exists(positive_path):
            raise FileNotFoundError(f"Scaler file not found : {positive_path}")
        if not os.path.exists(posi_nega_path):
            raise FileNotFoundError(f"Target Scaler file not found : {posi_nega_path}")    
        if not os.path.exists(target_power_path):
            raise FileNotFoundError(f"Target Scaler file not found : {target_power_path}")    
        
        self.positive_scaler = joblib.load(positive_path)
        self.posi_nega_scaler = joblib.load(posi_nega_path)
        self.power_scaler = joblib.load(target_power_path)