import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

from dataset import DrivingDataset
from model import LSTMModel, GRUModel, TransformerModel, BasicLSTMModel, TCNModel, NLinear, PatchTST
from evaluate import Evaluate
from utils.processor import DataNormalizer
from utils.loss_function import GradientLoss, TimeWeightedGradientLoss, WeightedPhysicsLoss, WeightedPhysicsLoss_ver2
from test import test_main

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device your using: {device}")

# BATCH_SIZE = 4096
# LEARNING_RATE = 0.1
# EPOCHS = 100   
# SEQ_LEN = 150  
# HORIZON = 50       
# INPUT_SIZE = 11      
# HIDDEN_SIZE = 128    
# NUM_LAYERS = 3     
# OUTPUT_SIZE = 50  

BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 1000    
SEQ_LEN = 150  
HORIZON = 50       
INPUT_SIZE = 11      
HIDDEN_SIZE = 64    
NUM_LAYERS = 1 
OUTPUT_SIZE = 50  

ALPHA = 0.8

PATIENCE = 15

# Loss
min_delta = 1e-6

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def prepare_data():
    print("Data load...")
    train_set = np.load("segmented_data/train_segments.npy", allow_pickle=True)
    valid_set = np.load("segmented_data/valid_segments.npy", allow_pickle=True)

    normalizer = DataNormalizer()

    normalizer.load_scaler("segmented_data/positive_scaler.pkl", "segmented_data/posi_nega_scaler.pkl", "segmented_data/power_scaler.pkl")

    return train_set, valid_set, normalizer

def save_attention_map(input_data, attn_weights, epoch, batch_idx, save_path="./results/plots"):
    # 배치 중 첫 번째 샘플만 시각화 (Batch, Seq, Feat) -> (Seq, Feat)
    sample_input = input_data[:, 7].cpu().detach().numpy()
    # (Batch, Seq, 1) -> (Seq,)
    sample_weights = attn_weights.cpu().detach().numpy()

    time_steps = np.linspace(0, 15, 150) # 0초부터 15초까지
    
    plt.figure(figsize=(15, 6))
    
    # 1. 상단: Attention Heatmap
    plt.subplot(2, 1, 1)
    # 히트맵을 2D로 만들기 위해 reshape (1, 150)
    sns.heatmap(sample_weights.reshape(1, -1), cmap='viridis', cbar=True)
    plt.title(f"Epoch {epoch+1} - Attention Weights (Past 15s)")
    plt.xlabel("Time Steps")
    
    # 2. 하단: 입력 피처와 가중치 비교 (이중 Y축 사용)
    ax1 = plt.subplot(2, 1, 2)
    
    # Input feature 그리기
    color_feat = 'tab:blue'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Input Feature (Normalized)', color=color_feat)
    ax1.plot(time_steps, sample_input, color=color_feat, alpha=0.6, label='Input Feature (Index 0)')
    ax1.tick_params(axis='y', labelcolor=color_feat)
    
    # 어텐션 스코어를 위한 두 번째 Y축
    ax2 = ax1.twinx()
    color_attn = 'tab:red'
    ax2.set_ylabel('Attention Score', color=color_attn)
    ax2.plot(time_steps, sample_weights, color=color_attn, linewidth=2, label='Attention Score')
    ax2.tick_params(axis='y', labelcolor=color_attn)
    
    # 가중치가 높은 시점에 수직선 표시 (Top 5 시점)
    top_indices = np.argsort(sample_weights)[-5:]
    for idx in top_indices:
        plt.axvline(x=time_steps[idx], color='orange', linestyle='--', alpha=0.3)

    plt.title(f"Attention Score vs Input Feature Comparison")
    plt.tight_layout()
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    plt.savefig(f"{save_path}/attn_epoch_{epoch+1}_idx_{batch_idx}.png")
    plt.close()

def plot_loss_contributions(loss_history, save_path="./results/plot_weights"):
    """
    loss_history: 각 에폭별 loss_dict가 담긴 리스트
    """
    df = pd.DataFrame(loss_history)
    epochs = range(1, len(df) + 1)

    plt.figure(figsize=(15, 6))

    # 1. 절대적 수치 변화 (Line Chart)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['weighted_data_loss'], label='Data Loss (Weighted)')
    plt.plot(epochs, df['weighted_physics_loss'], label='Physics Loss (Weighted)')
    plt.plot(epochs, df['weighted_torque_loss'], label='Torque Loss (Weighted)')
    plt.plot(epochs, df['weighted_rpm_loss'], label='RPM Loss (Weighted)')
    plt.plot(epochs, df['raw_mse'], label='Raw MSE')
    plt.title('Loss Components Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    # 2. 상대적 비중 분석 (Stacked Area Chart)
    # 전체 Loss 대비 각 성분이 차지하는 %를 계산
    plt.subplot(1, 2, 2)
    total = df['weighted_data_loss'] + df['weighted_physics_loss'] + df['weighted_torque_loss'] + df['weighted_rpm_loss']
    
    plt.stackplot(epochs, 
                  df['weighted_data_loss'] / total * 100,
                  df['weighted_physics_loss'] / total * 100,
                  df['weighted_torque_loss'] / total * 100,
                  df['weighted_rpm_loss'] / total * 100,
                  labels=['Data', 'Physics', 'Torque', 'RPM'],
                  alpha=0.7)
    
    plt.title('Loss Contribution Ratio (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage (%)')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()

    if not os.path.exists(save_path): os.makedirs(save_path)
    plt.savefig(f"{save_path}/weights.png")
    plt.close()

def plot_gradient_norm(data_norms, physics_norms, torque_norms, rpm_norms, save_path="./results/plot_norms"):
    """
    각 에포크별 그래디언트 노름(Gradient Norm) 변화를 시각화합니다.
    """
    epochs = range(1, len(data_norms) + 1)
    
    plt.figure(figsize=(15, 6))

    # 1. 절대적 수치 변화 (Line Chart - Log Scale)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data_norms, label='Data Grad Norm', marker='o', markersize=3)
    plt.plot(epochs, physics_norms, label='Physics Grad Norm', marker='s', markersize=3)
    plt.plot(epochs, torque_norms, label='Torque Grad Norm', alpha=0.7)
    plt.plot(epochs, rpm_norms, label='RPM Grad Norm', alpha=0.7)
    
    # 그래디언트 노름은 값의 범위 차이가 클 수 있으므로 로그 스케일 권장
    plt.yscale('log')
    plt.title('Gradient Norms Trend (Log Scale)')
    plt.xlabel('Epochs')
    plt.ylabel('Global Norm Value (Log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # 2. 상대적 기여도 비중 (Stacked Area Chart)
    plt.subplot(1, 2, 2)
    # 리스트들을 넘파이 배열로 변환하여 합계 계산
    norms_array = np.array([data_norms, physics_norms, torque_norms, rpm_norms])
    total_norms = np.sum(norms_array, axis=0)
    
    plt.stackplot(epochs, 
                  data_norms / total_norms * 100,
                  physics_norms / total_norms * 100,
                  torque_norms / total_norms * 100,
                  rpm_norms / total_norms * 100,
                  labels=['Data', 'Physics', 'Torque', 'RPM'],
                  alpha=0.7, colors=sns.color_palette("viridis", 4))
    
    plt.title('Gradient Contribution Ratio (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Update Energy Percentage (%)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    
    plt.savefig(f"{save_path}/gradient_norms.png")
    plt.close()
    # print(f"[Visualizer] Gradient norm plot saved to {save_path}")

def main():
    set_seed(42)

    train_set, valid_set, normalizer = prepare_data()

    # Apply sliding window
    train_dataset = DrivingDataset(segments=train_set, seq_len=SEQ_LEN, horizon=HORIZON, input_size=INPUT_SIZE)
    valid_dataset = DrivingDataset(segments=valid_set, seq_len=SEQ_LEN, horizon=HORIZON, input_size=INPUT_SIZE)
    
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=14, pin_memory=True, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    # Model Init (LSTM)
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, num_vars=3).to(device)

    #model = BasicLSTMModel(INPUT_SIZE, 128, 2, OUTPUT_SIZE, num_vars=3).to(device)

    # model = TCNModel(INPUT_SIZE, [66, 66, 66, 66, 66, 66], OUTPUT_SIZE, num_vars=3).to(device)

    # model = NLinear(SEQ_LEN, OUTPUT_SIZE, INPUT_SIZE).to(device)

    # model = PatchTST(SEQ_LEN, OUTPUT_SIZE).to(device)

    print(f"\n!!!Pleas check!!!, \n!!!Current Model: {model.__class__.__name__}!!!")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=9, threshold=0.0
    )

    gamma = 0.08
    beta = 0.15
    time_penalty = 10
    criterion = WeightedPhysicsLoss(
        alpha=ALPHA, 
        gamma=gamma, 
        beta=beta,
        time_penalty=time_penalty, 
        normalizer=normalizer, 
        horizon=HORIZON
    ).to(device)
    # criterion = nn.MSELoss()

    # Training Loop
    print("\n=== Train start ===")

    train_losses = []
    val_losses = []
    loss_history = []

    best_val_loss = float('inf')
    counter = 0
    
    data_grad_norm_list = []
    physisc_grad_norm_list = []
    torque_grad_norm_list = []
    rpm_grad_norm_list = []

    curriculum_flag = False

    for epoch in range(EPOCHS):
        epoch_loss_stats = None

        # Training
        model.train()
        train_loss = 0.0

        batch_data_grad_norms = []
        batch_physics_grad_norms = []
        batch_torque_grad_norms = []
        batch_rpm_grad_norms = []
        
        for batch_idx, (batch_X, batch_y, batch_y_input) in enumerate(train_loader):
            # torch.Size([128, 300, 9]) torch.Size([128, 50, 2]) torch.Size([128, 50, 9])
            #print(batch_X.size(), batch_y.size(), batch_y_input.size())
            batch_X, batch_y, batch_y_input = batch_X.to(device), batch_y.to(device), batch_y_input.to(device)
            
            # Forward
            prediction, _ = model(batch_X)

            # Loss
            loss, _, loss_dict = criterion(prediction, batch_y, batch_y_input)
            if epoch_loss_stats is None:
                epoch_loss_stats = {k: 0.0 for k in loss_dict.keys()}

            for k, v in loss_dict.items(): 
                epoch_loss_stats[k] += v.item()

            # Backward
            optimizer.zero_grad() 

            # if batch_idx % 10 == 0:
            #     ################## Gradient Norm (Data loss) ##################
            #     loss_dict['weighted_data_loss'].backward(retain_graph=True)
            #     data_grad_norm = 0.0
            #     for p in model.parameters():
            #         if p.grad is not None:
            #             data_grad_norm += p.grad.data.norm(2).item() ** 2
            #     data_grad_norm = data_grad_norm ** 0.5
            #     batch_data_grad_norms.append(data_grad_norm)

            #     optimizer.zero_grad()
            #     ################## Gradient Norm (Data loss) END ##################


            #     ################## Gradient Norm (Physical loss) ##################
            #     loss_dict['weighted_physics_loss'].backward(retain_graph=True)
            #     physics_grad_norm = 0.0
            #     for p in model.parameters():
            #         if p.grad is not None:
            #             physics_grad_norm += p.grad.data.norm(2).item() ** 2
            #     physics_grad_norm = physics_grad_norm ** 0.5
            #     batch_physics_grad_norms.append(physics_grad_norm)

            #     optimizer.zero_grad()
            #     ################## Gradient Norm (Physical loss) END ##################


            #     ################## Gradient Norm (Torque loss) ##################
            #     loss_dict['weighted_torque_loss'].backward(retain_graph=True)
            #     torque_grad_norm = 0.0
            #     for p in model.parameters():
            #         if p.grad is not None:
            #             torque_grad_norm += p.grad.data.norm(2).item() ** 2
            #     torque_grad_norm = torque_grad_norm ** 0.5
            #     batch_torque_grad_norms.append(torque_grad_norm)

            #     optimizer.zero_grad()
            #     ################## Gradient Norm (Torque loss) END ##################


            #     ################## Gradient Norm (RPM loss) ##################
            #     loss_dict['weighted_rpm_loss'].backward(retain_graph=True)
            #     rpm_grad_norm = 0.0
            #     for p in model.parameters():
            #         if p.grad is not None:
            #             rpm_grad_norm += p.grad.data.norm(2).item() ** 2
            #     rpm_grad_norm = rpm_grad_norm ** 0.5
            #     batch_rpm_grad_norms.append(rpm_grad_norm)

            #     optimizer.zero_grad()
            #     ################## Gradient Norm (RPM loss) END ##################

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)       
            optimizer.step()      

            train_loss += loss.item()

        if not batch_data_grad_norms:
            pass
        else:
            data_grad_norm_list.append(np.mean(batch_data_grad_norms))
            physisc_grad_norm_list.append(np.mean(batch_physics_grad_norms))
            torque_grad_norm_list.append(np.mean(batch_torque_grad_norms))
            rpm_grad_norm_list.append(np.mean(batch_rpm_grad_norms))
        
        avg_train_loss = train_loss / len(train_loader)

        if epoch_loss_stats is not None:
            avg_epoch_stats = {
                k: (v.item() if torch.is_tensor(v) else v) / len(train_loader) 
                for k, v in epoch_loss_stats.items()
            }
            loss_history.append(avg_epoch_stats)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y, batch_y_input) in enumerate(val_loader):
                batch_X, batch_y, batch_y_input = batch_X.to(device), batch_y.to(device), batch_y_input.to(device)

                if model.__class__.__name__ == "LSTMModel":
                    prediction, attn_weights = model(batch_X)
                    if batch_idx == 0:
                        save_attention_map(
                            input_data=batch_X[7],       
                            attn_weights=attn_weights[0],
                            epoch=epoch,
                            batch_idx=batch_idx
                        )
                else:
                    prediction, _ = model(batch_X)

                _, loss, _ = criterion(prediction, batch_y, batch_y_input)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # If loss changes lower than "min_delta", than add counter
        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            counter = 0
            
            # Save results
            if not os.path.exists('results'):
                os.makedirs('results')
            # torch.save(model.state_dict(), f"results/best_model_{criterion.time_penalty:.2f}.pth")
            torch.save(model.state_dict(), f"results/best_model.pth")
            print(f"***Best Model Saved!*** (Loss: {best_val_loss:.6f})")

            improved = True
        else:
            counter += 1 
            improved = False

        # Scheduler update & Logging
        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss) 
        after_lr = optimizer.param_groups[0]['lr']

        if not improved:
            state = scheduler.state_dict()
            sched_bad = state.get('num_bad_epochs', 'N/A')
            sched_patience = scheduler.patience
            print(f"--> No Improvement. Counter: {counter}/{PATIENCE} | Sched: Bad={sched_bad}/{sched_patience}")
        
        if before_lr != after_lr:
            print(f"Epoch [{epoch+1}] Learning Rate Reduced: {before_lr} -> {after_lr}")

        # Record loss
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early Stopping Trigger
        if counter >= PATIENCE:
            print("Early Stopping triggered! End training early")
            break

        if not rpm_grad_norm_list:
            pass
        else:
            plot_gradient_norm(data_grad_norm_list, physisc_grad_norm_list, torque_grad_norm_list, rpm_grad_norm_list)

    print(f"=== Training finished | Best raw MSE : {best_val_loss} ===")

    if len(loss_history) > 0:
        plot_loss_contributions(loss_history)

    evaluator = Evaluate(f"results/best_model.pth", f"results")
    evaluator.final_evaluation(model, val_loader, normalizer, device)
    #evaluator.plot_multistep_prediction(model, val_loader, normalizer, device, SEQ_LEN)
    #evaluator.error_by_horizon(model, val_loader, normalizer, device)
    evaluator.analyze_phase_lag(model, val_loader, normalizer, device)

    # Save Last Model
    if not os.path.exists('results'):
        os.makedirs('results')
        
    save_path = "results/lstm_ev_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Last Model saved successfully: {save_path}")

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig('results/loss_graph.png')
    
    print(f"\n!!!Check again!!!, \n!!!Used Model: {model.__class__.__name__}!!!\n")

    test_main(["results"], f"{model.__class__.__name__}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    elapsed = end_time - start_time

    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60

    print(
        f"Total processing time : {int(hours)}H : {int(minutes)}M : {seconds:.2f}S"
    )