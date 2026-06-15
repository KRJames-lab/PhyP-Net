from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import signal

class Evaluate:
    def __init__(self, checkpoint, save_path):
        self.checkpoint = checkpoint
        self.save_path = save_path

        self.log_file = os.path.join(save_path, "output.txt")

    def final_evaluation(self, model, val_loader, normalizer, device):
        print("\n=== Final Evaluation (with Best Model) ===")

        # Load best model
        if os.path.exists(self.checkpoint):
            model.load_state_dict(torch.load(self.checkpoint))
            print("-> Best model loaded.")
        else:
            print("Warning : Best model not found. Using current state.")

        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y, batch_y_input in val_loader:
                batch_X = batch_X.to(device)
                pred, _ = model(batch_X)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())

        # Convert list to numpy array
        full_preds = np.vstack(all_preds)   # (Total_Samples, 50, 2)
        full_targets = np.vstack(all_targets) # (Total_Samples, 50, 2)

        # 2. 전력(index 0) 데이터만 추출합니다. (Total_Samples, 50)
        np_pred_power = full_preds[:, :, 0]
        np_target_power = full_targets[:, :, 0]

        # Inverse transform : 0~1 -> Real value
        real_pred = normalizer.power_scaler.inverse_transform(np_pred_power)
        real_target = normalizer.power_scaler.inverse_transform(np_target_power)

        mse_0to1 = mean_squared_error(real_target[:, :10], real_pred[:, :10])
        rmse_0to1 = np.sqrt(mse_0to1)
        mae_0to1 = mean_absolute_error(real_target[:, :10], real_pred[:, :10])
        r2_0to1 = r2_score(real_target[:, :10], real_pred[:, :10])

        mse_1to3 = mean_squared_error(real_target[:, 10:30], real_pred[:, 10:30])
        rmse_1to3 = np.sqrt(mse_1to3)
        mae_1to3 = mean_absolute_error(real_target[:, 10:30], real_pred[:, 10:30])
        r2_1to3 = r2_score(real_target[:, 10:30], real_pred[:, 10:30])
        
        mse_3to5 = mean_squared_error(real_target[:, 30:], real_pred[:, 30:])
        rmse_3to5 = np.sqrt(mse_3to5)
        mae_3to5 = mean_absolute_error(real_target[:, 30:], real_pred[:, 30:])
        r2_3to5 = r2_score(real_target[:, 30:], real_pred[:, 30:])

        txt_results = (
            f"0s to 1s metrics\n"
            f"RMSE (0~1s) : {rmse_0to1/1000:.4f} (kW)\n"
            f"MAE (0~1s)  : {mae_0to1/1000:.4f} (kW)\n"
            f"R2  (0~1s)  : {r2_0to1:.4f}\n\n"

            f"1s to 3s metrics\n"
            f"RMSE (1~3s) : {rmse_1to3/1000:.4f} (kW)\n"
            f"MAE (1~3s)  : {mae_1to3/1000:.4f} (kW)\n"
            f"R2  (1~3s)  : {r2_1to3:.4f}\n\n"

            f"3s to 5s metrics\n"
            f"RMSE (3~5s) : {rmse_3to5/1000:.4f} (kW)\n"
            f"MAE (3~5s)  : {mae_3to5/1000:.4f} (kW)\n"
            f"R2  (3~5s)  : {r2_3to5:.4f}\n"
        )
        print(txt_results)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(txt_results + f"{'='*30}\n")

        # Display only first steps
        plt.figure(figsize=(15, 6))
        
        # Show only part of one
        subset_target = real_target[:600, 0] / 1000
        subset_pred = real_pred[:600, 0] / 1000    
        
        plt.plot(subset_target, label='Actual Power', color='blue', alpha=0.7)
        plt.plot(subset_pred, label='Predicted Power', color='red', linestyle='--', alpha=0.7)
        
        plt.title('Prediction vs Actual (1st step only)')
        plt.xlabel('Time Steps (0.1s)')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/prediction_comparison.png')
        plt.close()
        print(f"-> Prediction graph saved to {self.save_path}/prediction_comparison.png")

    def plot_multistep_prediction(self, model, val_loader, normalizer, device, SEQ_LEN=100):
        print("\n=== Plot multistep prediction (Snapshot) ===")

        if os.path.exists(self.checkpoint):
            model.load_state_dict(torch.load(self.checkpoint))

        model.eval()

        # Brings only one batch
        batch_X, batch_y, batch_y_input = next(iter(val_loader))
        batch_X = batch_X.to(device)

        # predict
        with torch.no_grad():
            pred, _ = model(batch_X)
        
        # Inverse transform
        real_pred = normalizer.power_scaler.inverse_transform(pred.cpu().numpy())
        real_target = normalizer.power_scaler.inverse_transform(batch_y.numpy())

        plt.figure(figsize=(15, 6))

        # Draw Ground Truth
        ground_truth_line = real_target[:300, 0] / 1000 

        plt.plot(range(SEQ_LEN, SEQ_LEN + 300), ground_truth_line, 
                    label='Ground Truth', color='gray', alpha=0.4, linewidth=0.8)

        # Snapshot (Red Lines)
        for i in range(0, 300, 50): 
            future_pred = real_pred[i] / 1000

            start_time = i + SEQ_LEN
            time_steps = np.arange(start_time, start_time + 50)

            plt.plot(time_steps, future_pred, color='red', alpha=0.8, linewidth=1.5)
            plt.scatter(start_time, future_pred[0], color='red', s=10)

        # Draw a vertical line every 10 seconds (100 steps)
        for x_line in range(SEQ_LEN, SEQ_LEN + 300, 100):
            plt.axvline(x=x_line, color='gray', linestyle=':', alpha=0.5)
            plt.text(x_line, ground_truth_line.max(), f'{(x_line-SEQ_LEN)/10}s', ha='center')

        plt.title('Multi-step Prediction')

        plt.xlabel('Time Steps (0.1s)') 
        plt.ylabel('Power (kW)')

        plt.legend(['Ground Truth', 'Predicted Horizon (5s)'])

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/multistep_snapshot.png')
        plt.close()
        print(f"-> Snapshot graph saved to {self.save_path}/multistep_snapshot.png")

    def error_by_horizon(self, model, val_loader, normalizer, device):
        print("\n=== Processing error by horizon ===")
        
        if os.path.exists(self.checkpoint):
            model.load_state_dict(torch.load(self.checkpoint))

        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y, batch_y_input in val_loader:
                batch_X = batch_X.to(device)
                pred, _ = model(batch_X)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())

        np_preds = np.concatenate(all_preds, axis=0)
        np_target = np.concatenate(all_targets, axis=0)

        real_pred = normalizer.power_scaler.inverse_transform(np_preds) / 1000
        real_target = normalizer.power_scaler.inverse_transform(np_target) / 1000

        mse = np.mean((real_target - real_pred) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(real_target - real_pred), axis=0)
        r2 = r2_score(real_target, real_pred)

        fig, axes = plt.subplots(1, 3, figsize=(15,6))

        time_steps = np.arange(1, 51) * 0.1
        
        axes[0].plot(time_steps, mse)
        axes[0].set_title("MSE over Horizon")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("MSE (kW)")

        axes[1].plot(time_steps, rmse)
        axes[1].set_title("RMSE over Horizon")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("RMSE (kW)")

        axes[2].plot(time_steps, mae)
        axes[2].set_title("MAE over Horizon")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("MAE (kW)")

        plt.tight_layout()

        fig.savefig(f'{self.save_path}/error_by_horizon.png')
        plt.close()
        txt_results = (
            f"RMSE at 0.1s: {rmse[0]:.4f} kW \n"
            f"RMSE at 5.0s: {rmse[-1]:.4f} kW \n"
        )
        print(txt_results)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(txt_results + f"{'='*30}\n")
    
    def analyze_phase_lag(self, model, val_loader, normalizer, device):
        print("\n=== Start analyzing phase lag ===")

        if os.path.exists(self.checkpoint):
            model.load_state_dict(torch.load(self.checkpoint))

        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y, batch_y_input in val_loader:
                batch_X = batch_X.to(device)
                pred, _ = model(batch_X)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())

        # Convert list to numpy array
        full_preds = np.vstack(all_preds)   # (Total_Samples, 50, 2)
        full_targets = np.vstack(all_targets) # (Total_Samples, 50, 2)

        # 2. 전력(index 0) 데이터만 추출합니다. (Total_Samples, 50)
        np_pred_power = full_preds[:, :, 0]
        np_target_power = full_targets[:, :, 0]

        # Inverse transform : 0~1 -> Real value
        real_pred = normalizer.power_scaler.inverse_transform(np_pred_power) / 1000
        real_target = normalizer.power_scaler.inverse_transform(np_target_power) / 1000

        x_sig = real_target[:600, 0]
        x_sig = x_sig - np.mean(x_sig)

        y_sig = real_pred[:600, 0]
        y_sig = y_sig - np.mean(y_sig)

        correlation = signal.correlate(x_sig, y_sig, mode='full')
        lags = signal.correlation_lags(len(x_sig), len(y_sig), mode='full')

        # 4. Normalization (-1 ~ 1 scaling)
        # Pearson Correlation Coefficient와 동일한 스케일로 변환
        normalization_factor = np.sqrt(np.sum(x_sig**2) * np.sum(y_sig**2))
        if normalization_factor == 0:
            norm_correlation = correlation # 분모가 0인 경우 예외처리
        else:
            norm_correlation = correlation / normalization_factor

        # 5. Find Max Lag
        max_idx = np.argmax(norm_correlation)
        lag_at_max = lags[max_idx]
        max_corr_val = norm_correlation[max_idx]
        
        if max_idx != 0:
            zero_lag_idx = np.where(lags == 0)[0]
            val_at_zero_lag = norm_correlation[zero_lag_idx[0]]

        txt_results = (
            f'[Result] Max Correlation: {max_corr_val:.4f} at Lag: {lag_at_max} steps'
            f'[Result] Correlation at Lag 0: {val_at_zero_lag:.4f} \n'
        )
        print(txt_results)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(txt_results + f"{'='*30}\n")

        plt.figure(figsize=(15, 6))
        
        plt.plot(lags, norm_correlation, label='Cross Correlation', color='blue', linewidth=1.5)
        
        plt.axvline(0, color='black', linestyle='--', alpha=0.5, label='Zero Lag')
        plt.axvline(lag_at_max, color='red', linestyle='--', alpha=0.8, label=f'Max Lag ({lag_at_max})')
        
        # Max Point 마커 표시
        plt.scatter(lag_at_max, max_corr_val, color='red', zorder=5)
        
        # Labels & Title (Formal Terminology)
        plt.title(f"Time-lag Cross Correlation Analysis\n(Max Lag: {lag_at_max} steps, Coeff: {max_corr_val:.2f})")
        plt.xlabel("Time Lag (Steps)")
        plt.ylabel("Normalized Cross-Correlation Coefficient")
        
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.tight_layout()

        lag_save_path = f'{self.save_path}/correlation_lag_analysis.png'
        plt.savefig(lag_save_path)
        plt.close()
        print(f"Graph saved to {lag_save_path}\n")