import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from unittest.mock import patch
from scipy.ndimage import binary_dilation, label

from utils.processor import DataNormalizer
from utils.reshape_data import padding

from model import LSTMModel, GRUModel, TransformerModel, BasicLSTMModel, TCNModel, NLinear, PatchTST

TARGET_PATH = [
    "results_BasicLSTMModel"
    ]

# CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Test Device: {device}")

# Must use same structure with trained model
BATCH_SIZE = 16384
LEARNING_RATE = 0.001
EPOCHS = 1000    
SEQ_LEN = 150  
HORIZON = 50       
INPUT_SIZE = 11      
HIDDEN_SIZE = 64   
NUM_LAYERS = 1
OUTPUT_SIZE = 50  

# Transformer Params
D_MODEL = 64
NHEAD = 4

ALPHA = 0.8

# Test data root
TEST_ROOT = "segmented_data/test_segments.npy"

class TestEvaluation():
    def __init__(self, model, data_path, normalizer, device, path):
        self.model = model
        self.data_path = data_path
        self.normalizer = normalizer
        self.device = device
        self.target_path = path

        self.efficiency = 0.97

        self.drive_list, self.all_features_drive_list = self._run_sliding_window_inference()

    # Sliding window inference method (batched for speed)
    def _run_sliding_window_inference(self, batch_size=4096):
        if os.path.exists(self.data_path):
            test_driving_set = np.load(self.data_path, allow_pickle=True)

        self.model.eval()

        num_segments = len(test_driving_set)
        n_features = self.normalizer.posi_nega_scaler.n_features_in_

        drive_list = []
        all_features_drive_list = []

        for i in range(num_segments):
            current_segment = test_driving_set[i]
            max_t = len(current_segment) - SEQ_LEN - HORIZON + 1
            print(f"\nDrive{i+1} inference starts... ({max_t} windows)")

            # ── Collect all targets (no GPU needed) ──
            all_target_power = np.zeros((max_t, HORIZON, 1))
            all_target_all = np.zeros((max_t, HORIZON, 3))

            for t in range(max_t):
                window_y = current_segment[t+SEQ_LEN : t+SEQ_LEN+HORIZON, INPUT_SIZE:]
                # Power target
                all_target_power[t] = self.normalizer.power_scaler.inverse_transform(
                    window_y[:, 0].reshape(-1, 1))
                # All features target
                temp_buffer = np.zeros((HORIZON, n_features))
                temp_buffer[:, -3:] = window_y
                inv_result = self.normalizer.posi_nega_scaler.inverse_transform(temp_buffer)[:, -3:]
                inv_result[:, 0:1] = all_target_power[t]
                all_target_all[t] = inv_result

            # ── Batched model inference ──
            all_pred_power = np.zeros((max_t, HORIZON, 1))
            all_pred_all = np.zeros((max_t, HORIZON, 3))

            for start in range(0, max_t, batch_size):
                end = min(start + batch_size, max_t)
                # Build batch
                batch_X = np.array([
                    current_segment[t : t+SEQ_LEN, :INPUT_SIZE]
                    for t in range(start, end)
                ])
                batch_X_tensor = torch.from_numpy(batch_X).float().to(self.device)

                with torch.no_grad():
                    batch_pred, _ = self.model(batch_X_tensor)

                batch_pred_np = batch_pred.cpu().numpy()  # (B, 50, 3)

                # Power inverse transform
                pred_power = batch_pred_np[:, :, 0].reshape(-1, 1)
                pred_power = self.normalizer.power_scaler.inverse_transform(pred_power)
                pred_power = pred_power.reshape(end - start, HORIZON, 1)
                all_pred_power[start:end] = pred_power

                # All features inverse transform
                B = end - start
                pred_3ch = batch_pred_np.reshape(B * HORIZON, 3)
                temp_buffer = np.zeros((B * HORIZON, n_features))
                temp_buffer[:, -3:] = pred_3ch
                inv_all = self.normalizer.posi_nega_scaler.inverse_transform(temp_buffer)[:, -3:]
                inv_all = inv_all.reshape(B, HORIZON, 3)
                inv_all[:, :, 0:1] = pred_power
                all_pred_all[start:end] = inv_all

            drive_list.append({
                "segment_index": i,
                "target": all_target_power,       # (n, 50, 1)
                "prediction": all_pred_power
            })

            all_features_drive_list.append({
                "segment_index": i,
                "target": all_target_all,          # (n, 50, 3)
                "prediction": all_pred_all
            })

            print(f"Infering Drive{i+1} has done")
        print("\n=== All drive inferred well ===")

        return drive_list, all_features_drive_list
    
    def plot_first_step_only(self):
        print("\n === Comparing only first step === ")

        save_path = f'{self.target_path}/tests'
        os.makedirs(save_path, exist_ok=True)
        
        test_length = 2000
        
        for i, data in enumerate(self.drive_list):
            print(f"Drive{i+1} now processing...")

            # (19803, 50, 1) -> (19803, 50)
            squeezed_target = data["target"].squeeze()
            squeezed_pred = data["prediction"].squeeze()

            # Show only part of one
            subset_target = squeezed_target[:test_length, 0] / 1000
            subset_pred = squeezed_pred[:test_length, 0] / 1000    
            
            # Display only first steps
            plt.figure(figsize=(15, 6))
            plt.xlim(SEQ_LEN, test_length+SEQ_LEN)

            plt.plot(range(SEQ_LEN, test_length+SEQ_LEN), subset_target, label='Actual Power', color='blue', alpha=0.7)
            plt.plot(range(SEQ_LEN, test_length+SEQ_LEN), subset_pred, label='Predicted Power', color='red', linestyle='--', alpha=0.7)

            plt.title(f'Drive{i+1} : Prediction vs Actual (1st step only)')
            plt.xlabel('Time Steps (0.1s)')
            plt.ylabel('Power (kW)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.close()
            os.makedirs(os.path.join(save_path, f"Drive{i+1}"), exist_ok=True)
            plt.savefig(os.path.join(save_path, f"Drive{i+1}", f"Drive{i+1} - First_step_comparison.png"))
            print(f"-> Prediction graph saved to results/Drive{i+1}/Drive{i+1} - First_step_comparison.png")
        
        print("\nComparing only first step has done\n")

    def plot_multistep_snapshot(self):
        print("=== Plot multistep snapshot ===")
        
        save_path = f'{self.target_path}/tests'
        os.makedirs(save_path, exist_ok=True)

        # Set graph range
        start_x = 1000
        end_x = 1400

        for i, data in enumerate(self.drive_list):
            print(f"Drive{i+1} now processing...")

            plt.figure(figsize=(15, 6))
            plt.xlim(start_x, end_x)
            plt.title(f"Drive{i+1} - Multi-step Prediction")

            squeezed_target = data["target"].squeeze()
            squeezed_pred = data["prediction"].squeeze()

            for j in range(start_x, end_x, OUTPUT_SIZE):
                plt.plot(range(j, j+OUTPUT_SIZE), squeezed_target[j - OUTPUT_SIZE, :]/1000, label='Ground Truth', color='gray', alpha=0.6, linewidth=0.8)
                plt.plot(range(j, j+OUTPUT_SIZE), squeezed_pred[j - OUTPUT_SIZE, :]/1000, color='red', alpha=0.8, linewidth=1.5)
                plt.axvline(x=j, color='gray', alpha=0.9)

            plt.xlabel('Time Steps (0.1s)') 
            plt.ylabel('Power (kW)')
            plt.legend(['Ground Truth', 'Predicted Horizon'])
            plt.grid(True)
            plt.tight_layout()
            
            os.makedirs(os.path.join(save_path, f"Drive{i+1}"), exist_ok=True)
            plt.savefig(os.path.join(save_path, f"Drive{i+1}" , f"Drive{i+1} - Multi-step Prediction.png"))
            print(f"-> Prediction graph saved to results/Drive{i+1}/Drive{i+1} - Multi-step Prediction.png")

            plt.close()

        print("\nPlot multistep snapshot has done\n")

    def plot_error_by_horizon(self):
        print("=== Plot Error By Horizon ===")

        time_steps = np.arange(0, 5, 0.1)

        save_path = f'{self.target_path}/tests'
        os.makedirs(save_path, exist_ok=True)

        for i, data in enumerate(self.drive_list):
            print(f"Drive{i+1} now processing...")

            squeezed_target = data["target"].squeeze()
            squeezed_pred = data["prediction"].squeeze()

            mse_list = []
            rmse_list = []
            mae_list = []

            for j in range(OUTPUT_SIZE):
                mse = mean_squared_error(squeezed_target[:, j]/1000, squeezed_pred[:, j]/1000)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(squeezed_target[:, j]/1000, squeezed_pred[:, j]/1000)

                mse_list.append(mse)
                rmse_list.append(rmse)
                mae_list.append(mae)
            
            fig, axes = plt.subplots(1, 3, figsize=(15,6))
            fig.suptitle(f"Drive{i+1}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.99])

            axes[0].plot(time_steps, mse_list)
            axes[0].set_title(f" MSE over Horizon")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("MSE (kW)")

            axes[1].plot(time_steps, rmse_list)
            axes[1].set_title(f"RMSE over Horizon")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("RMSE (kW)")

            axes[2].plot(time_steps, mae_list)
            axes[2].set_title(f"MAE over Horizon")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("MAE (kW)")

            for ax in axes:
                ax.set_xlim(0, len(time_steps)/10)
            
            os.makedirs(os.path.join(save_path, f"Drive{i+1}"), exist_ok=True)
            plt.savefig(os.path.join(save_path, f"Drive{i+1}", f"Drive{i+1} - Error By Horizon.png"))
            print(f"-> Prediction graph saved to results/Drive{i+1}/Drive{i+1} - Error By Horizon.png")
            plt.close()
        print("\nPlot error by horizon has done\n")

    def show_evaluation_metrixs(self):
        print("\n=== Print evaluation metrixs ===\n")

        save_path = f'{self.target_path}/tests'
        os.makedirs(save_path, exist_ok=True)

        total_metrics = []

        for i, data in enumerate(self.drive_list):
            squeezed_target = data["target"].squeeze()
            squeezed_pred = data["prediction"].squeeze()

            mse_0to1 = mean_squared_error(squeezed_target[:, :10], squeezed_pred[:, :10])
            rmse_0to1 = np.sqrt(mse_0to1)
            mae_0to1 = mean_absolute_error(squeezed_target[:, :10], squeezed_pred[:, :10])
            r2_0to1 = r2_score(squeezed_target[:, :10], squeezed_pred[:, :10])

            mse_1to3 = mean_squared_error(squeezed_target[:, 10:30], squeezed_pred[:, 10:30])
            rmse_1to3 = np.sqrt(mse_1to3)
            mae_1to3 = mean_absolute_error(squeezed_target[:, 10:30], squeezed_pred[:, 10:30])
            r2_1to3 = r2_score(squeezed_target[:, 10:30], squeezed_pred[:, 10:30])

            mse_3to5 = mean_squared_error(squeezed_target[:, 30:], squeezed_pred[:, 30:])
            rmse_3to5 = np.sqrt(mse_3to5)
            mae_3to5 = mean_absolute_error(squeezed_target[:, 30:], squeezed_pred[:, 30:])
            r2_3to5 = r2_score(squeezed_target[:, 30:], squeezed_pred[:, 30:])         

            total_metrics.append([
                rmse_0to1, mae_0to1, r2_0to1,
                rmse_1to3, mae_1to3, r2_1to3,
                rmse_3to5, mae_3to5, r2_3to5
            ])

            text_file = f"{save_path}/Drive{i+1}/Drive{i+1} - Evaluation metrixs.txt"
            result_text = (
                f"*** Drive{i+1} evaluation metrixs ***\n"
                f"<0s to 1s>\n"
                f"RMSE (0~1s) : {rmse_0to1/1000:.4f} (kW)\n"
                f"MAE (0~1s)  : {mae_0to1/1000:.4f} (kW)\n"
                f"R2  (0~1s)  : {r2_0to1:.4f}\n\n"
                
                f"<1s to 3s>\n"
                f"RMSE (1~3s) : {rmse_1to3/1000:.4f} (kW)\n"
                f"MAE (1~3s)  : {mae_1to3/1000:.4f} (kW)\n"
                f"R2  (1~3s)  : {r2_1to3:.4f}\n\n"
                
                f"<3s to 5s>\n"
                f"RMSE (3~5s) : {rmse_3to5/1000:.4f} (kW)\n"
                f"MAE (3~5s)  : {mae_3to5/1000:.4f} (kW)\n"
                f"R2  (3~5s)  : {r2_3to5:.4f}\n"
            )
            print(result_text)
            
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(result_text)
            
        # Total metrics
        if total_metrics:
            avg_m = np.mean(total_metrics, axis=0)

            total_result_text = (
                f"\n*** Total Average Evaluation Metrics (All Drives) ***\n"
                f"<0s to 1s Average>\n"
                f"RMSE : {avg_m[0]/1000:.4f} (kW)\n"
                f"MAE  : {avg_m[1]/1000:.4f} (kW)\n"
                f"R2   : {avg_m[2]:.4f}\n\n"
                
                f"<1s to 3s Average>\n"
                f"RMSE : {avg_m[3]/1000:.4f} (kW)\n"
                f"MAE  : {avg_m[4]/1000:.4f} (kW)\n"
                f"R2   : {avg_m[5]:.4f}\n\n"
                
                f"<3s to 5s Average>\n"
                f"RMSE : {avg_m[6]/1000:.4f} (kW)\n"
                f"MAE  : {avg_m[7]/1000:.4f} (kW)\n"
                f"R2   : {avg_m[8]:.4f}\n"
            )

            print(total_result_text)

            total_file_path = f"{save_path}/Total - Evaluation metrixs.txt"
            with open(total_file_path, "w", encoding="utf-8") as f:
                f.write(total_result_text)

        print("\nPrint evaluation metrixs had donee\n")

    def get_peak_failure_rate(self, threshold=30000.0, rate=0.1, window_size=10):
        """
        주행 데이터 기반 과도 응답 추종 능력 및 정착 시간(Settling Time) 평가
        window_size: 과도 발생 후 관찰할 샘플 수 (예: 10Hz 데이터에서 10샘플 = 1초)
        """
        print("\n=== Calculating Peak Failure Rate & Settling Time (Driving Context) ===\n")
        
        save_path = f'{self.target_path}/tests'
        drive_results = []
        
        for i, data in enumerate(self.drive_list):
            squeezed_target = data["target"][:, 0].squeeze() 
            squeezed_pred = data["prediction"][:, 0].squeeze()
            
            # 1. 과도 현상 트리거 검출 (Slope Threshold)
            # 회생 제동(Regenerative Braking)을 포함한 급격한 변화 감지
            diff = np.abs(np.diff(squeezed_target, axis=-1))
            trigger_mask = diff > threshold
            
            # 2. 관찰 구간 확장 및 개별 이벤트 분리
            structure = np.ones(window_size + 1)
            transient_mask = binary_dilation(trigger_mask, structure=structure)
            
            # 개별적인 과도 이벤트 블록을 식별합니다. (Labeling)
            labeled_array, num_features = label(transient_mask)
            
            all_errors = []
            event_settling_times = []
            event_max_errors = []
            total_failures = 0
            total_transient_samples = 0

            # 3. 각 과도 이벤트 블록별 분석
            for feature_idx in range(1, num_features + 1):
                # 현재 이벤트 블록의 인덱스 추출
                event_indices = (labeled_array == feature_idx)
                
                # target/pred 정렬 (diff가 index 1부터이므로 보정)
                t_slice = squeezed_target[1:][event_indices]
                p_slice = squeezed_pred[1:][event_indices]
                
                if len(t_slice) == 0: continue

                # 오차 계산 (Relative Error)
                rel_errors = np.abs(t_slice - p_slice) / (np.abs(t_slice) + 1e-9)
                
                # 통계 데이터 축적
                total_failures += np.sum(rel_errors > rate)
                total_transient_samples += len(rel_errors)
                event_max_errors.append(np.max(rel_errors) * 100)
                
                # --- Settling Time(정착 시간) 계산 로직 ---
                # 역방향으로 탐색하여 처음으로 오차 기준(rate)을 벗어나는 지점을 찾음
                unstable_indices = np.where(rel_errors > rate)[0]
                if len(unstable_indices) == 0:
                    settling_time = 0  # 즉시 정착
                else:
                    # 마지막 불안정 지점 + 1이 정착 시점
                    settling_time = unstable_indices[-1] + 1
                
                event_settling_times.append(settling_time)

            # 4. 드라이브별 최종 지표 산출
            failure_rate = (total_failures / total_transient_samples * 100) if total_transient_samples > 0 else 0
            avg_settling_time = np.mean(event_settling_times) if event_settling_times else 0
            max_error_overall = np.max(event_max_errors) if event_max_errors else 0
            
            drive_results.append({
                'failure_rate': failure_rate,
                'settling_time': avg_settling_time,
                'max_error': max_error_overall
            })

            # 결과 출력 및 저장
            result_text = (
                f"\n<Drive{i+1} Transient Analysis>\n"
                f"Observation Window: {window_size} samples\n"
                f"Failure Rate: {failure_rate:.4f} (%)\n"
                f"Average Settling Time: {avg_settling_time:.2f} samples ({avg_settling_time/10:.2f}s)\n"
                f"Maximum Transient Error: {max_error_overall:.4f} (%)\n"
                f"Detected Transient Events: {num_features}\n"
                f"{'='*45}\n"
            )
            print(result_text)
            
            text_file = f"{save_path}/Drive{i+1}/Drive{i+1} - Evaluation metrixs.txt"
            with open(text_file, "a", encoding="utf-8") as f:
                f.write(result_text)

        # === 종합 성적 계산 ===
        if drive_results:
            avg_fail = np.mean([d['failure_rate'] for d in drive_results])
            avg_settle = np.mean([d['settling_time'] for d in drive_results])
            worst_err = np.max([d['max_error'] for d in drive_results])
            
            total_text = (
                f"\n<Total Performance Summary>\n"
                f"Average Failure Rate: {avg_fail:.4f} (%)\n"
                f"Global Average Settling Time: {avg_settle:.2f} samples\n"
                f"Worst Case Transient Error: {worst_err:.4f} (%)\n"
                f"{'='*45}\n"
            )
            print(total_text)
            with open(f"{save_path}/Total - Evaluation metrixs.txt", "a", encoding="utf-8") as f:
                f.write(total_text)

    def detect_violation(self, violation_threshold=1000.0):
        save_path = f'{self.target_path}/tests'
        os.makedirs(save_path, exist_ok=True)
        
        average_violations = []
        max_violations = []
        violation_counts = []
        for i, data in enumerate(self.all_features_drive_list):
            print(f"Drive{i+1} now processing...")

            # (19803, 50, 3) -> (19803, 50)
            power_target = data["target"][0].squeeze()
            power_pred = data["prediction"][0].squeeze()

            torque_target = data["target"][1].squeeze()
            torque_pred = data["prediction"][1].squeeze()

            rpm_target = data["target"][2].squeeze()
            rpm_pred = data["prediction"][2].squeeze()

            p_calc = torque_pred * (2 * np.pi * rpm_pred) / 60 * self.efficiency

            epsilon = abs(power_target - p_calc)

            average_violation = np.mean(epsilon)/1000
            max_violation = np.max(epsilon)/1000

            violation_count = np.sum(epsilon > violation_threshold)

            average_violations.append(average_violation)
            max_violations.append(max_violation)
            violation_counts.append(violation_count)

            text_file = f"{save_path}/Drive{i+1}/Drive{i+1} - Evaluation metrixs.txt"
            result_text = (
                f"\nAverage Violation : {average_violation:.2f}\n"
                f"Max Violation : {max_violation:.2f}\n"
                f"Violation count(Threshold(kW):{violation_threshold/1000}) : {violation_count}"
            )
            print(result_text)
            
            with open(text_file, "a", encoding="utf-8") as f:
                f.write(result_text)
        
        total_avg = np.mean(average_violations)
        total_max = np.mean(max_violations)
        total_count = sum(violation_counts)

        total_result_text = (
            f"\nTotal Average Violation : {total_avg:.2f}\n"
            f"Total Max Violation (Average) : {total_max:.2f}\n"
            f"Total violation count : {total_count}"
        )
        print(total_result_text)

        total_file_path = f"{save_path}/Total - Evaluation metrixs.txt"
        with open(total_file_path, "a", encoding="utf-8") as f:
            f.write(total_result_text)

# main
def test_main(target_path=TARGET_PATH, model_name="BasicLSTMModel", temperature=0.1):
    for path in target_path:
        # Load model and scaler
        print(f"Root : {path} | Load best model...")

        if model_name == "LSTMModel":
            model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, num_vars=3, temperature=temperature).to(device)
        elif model_name == "BasicLSTMModel":
            model = BasicLSTMModel(INPUT_SIZE, 96, 1, OUTPUT_SIZE, num_vars=3).to(device)
        elif model_name == "TCNModel":
            model = TCNModel(INPUT_SIZE, [32, 32, 32, 32, 32, 32], OUTPUT_SIZE, num_vars=3).to(device)
        elif model_name == "NLinear":
            model = NLinear(SEQ_LEN, OUTPUT_SIZE, INPUT_SIZE).to(device)
        elif model_name == "PatchTST":
            model = PatchTST(SEQ_LEN, OUTPUT_SIZE).to(device)
        else:
            print("No model name")
        
        model.load_state_dict(torch.load(os.path.join(path, "best_model.pth")))

        print("Done")

        print("\nLoad scaler...")
        normalizer = DataNormalizer()
        normalizer.load_scaler("segmented_data/positive_scaler.pkl", "segmented_data/posi_nega_scaler.pkl", "segmented_data/power_scaler.pkl")
        print("Done")

        print("\n=== Evaluation start!!! ===")
        test_evaluator = TestEvaluation(model, TEST_ROOT, normalizer, device, path)
        test_evaluator.plot_first_step_only()
        test_evaluator.plot_multistep_snapshot()
        test_evaluator.plot_error_by_horizon()
        test_evaluator.show_evaluation_metrixs()
        test_evaluator.get_peak_failure_rate()
        test_evaluator.detect_violation()

        # """ Dummy Test """
        # dummy = []
        # for i in range(5):
        #     dummy.append({
        #         "segment_index": i,
        #         "target": np.random.rand(100000, 50, 1),
        #         "prediction": np.random.rand(100000, 50, 1)
        #     })
        # dummy_obj = TestEvaluation.__new__(TestEvaluation)
        # dummy_obj.drive_list = dummy

        # # dummy_obj.plot_first_step_only()
        # # dummy_obj.plot_multistep_snapshot()
        # # dummy_obj.plot_error_by_horizon()
        # dummy_obj.show_evaluation_metrixs()
        # """ Dummy End """

        print(f"\n!!!Check!!!, \n!!!Used Model: {model.__class__.__name__}!!!\n")

if __name__ == "__main__":
    test_main(TARGET_PATH, "BasicLSTMModel")