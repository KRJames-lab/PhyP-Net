import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.processor import DataNormalizer

class GradientLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(GradientLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)

        delta_pred = pred[:, 1:] - pred[:, :-1]
        delta_target = target[:, 1:] - target[:, :-1]
        
        gradient_mse = self.mse(delta_pred, delta_target)

        return (self.alpha * mse) + ((1 - self.alpha) * gradient_mse), gradient_mse

class TimeWeightedGradientLoss(nn.Module):
    def __init__(self, alpha=0.5, time_penalty=1.0, horizon=50):
        super(TimeWeightedGradientLoss, self).__init__()
        self.alpha = alpha
        # time_penalty: 뒤쪽 스텝에 줄 추가 가중치 (0이면 가중치 없음)
        
        # 1. 값(Value)에 대한 시간 가중치 (50스텝)
        val_w = torch.linspace(1.0, 1.0 + time_penalty, steps=horizon)
        self.register_buffer('val_w', val_w)
        
        # 2. 변화량(Gradient)에 대한 시간 가중치 (49스텝)
        grad_w = torch.linspace(1.0, 1.0 + time_penalty, steps=horizon)
        self.register_buffer('grad_w', grad_w)

    def forward(self, pred, target):
        # --- 1. 기본 MSE 부분 (시간 가중치 적용) ---
        # (batch, 50) 형태의 제곱 오차에 가중치 곱하기
        mse_elementwise = (pred - target) ** 2
        weighted_mse = (mse_elementwise * self.val_w).mean()

        # --- 2. Gradient MSE 부분 (시간 가중치 적용) ---
        delta_pred = pred[:, 1:] - pred[:, :-1]
        delta_target = target[:, 1:] - target[:, :-1]
        
        grad_mse_elementwise = (delta_pred - delta_target) ** 2
        weighted_grad_mse = (grad_mse_elementwise * self.grad_w).mean()

        # 기존 alpha 로직 유지
        final_loss = (self.alpha * weighted_mse) + ((1 - self.alpha) * weighted_grad_mse)
        
        return final_loss, weighted_grad_mse

class IntegratedComparativeLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.05, loss_type='mse'):
        super(IntegratedComparativeLoss, self).__init__()
        self.alpha = alpha  # Value vs Gradient weight
        self.beta = beta    # Weight for Torque and RPM
        self.loss_type = loss_type.lower()

    def forward(self, pred, target):
        # 1. Component Extraction (자동 분리)
        # Power: [:, :, 0], Torque: [:, :, -2], RPM: [:, :, -1]
        components = {
            'p': (pred[:, :, 0], target[:, :, 0]),
            't': (pred[:, :, -2], target[:, :, -2]),
            'r': (pred[:, :, -1], target[:, :, -1])
        }

        # 2. 선택된 모드에 따른 손실 계산
        if self.loss_type == 'mse':
            return self._compute_mse(components)
        elif self.loss_type == 'mae':
            return self._compute_mae(components)
        elif self.loss_type == 'logcosh':
            return self._compute_logcosh(components)
        elif self.loss_type == 'sobolev':
            return self._compute_sobolev(components)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _compute_mse(self, comp):
        # MSE = mean((pred - target)^2)
        loss_p = F.mse_loss(comp['p'][0], comp['p'][1])
        loss_t = F.mse_loss(comp['t'][0], comp['t'][1])
        loss_r = F.mse_loss(comp['r'][0], comp['r'][1])
        return loss_p + self.beta * (loss_t + loss_r)

    def _compute_mae(self, comp):
        # MAE = mean(abs(pred - target))
        loss_p = F.l1_loss(comp['p'][0], comp['p'][1])
        loss_t = F.l1_loss(comp['t'][0], comp['t'][1])
        loss_r = F.l1_loss(comp['r'][0], comp['r'][1])
        return loss_p + self.beta * (loss_t + loss_r)

    def _compute_logcosh(self, comp):
        # Log-Cosh = log(cosh(x))
        def log_cosh(p, t):
            return torch.mean(torch.log(torch.cosh(p - t + 1e-12)))
        
        loss_p = log_cosh(*comp['p'])
        loss_t = log_cosh(*comp['t'])
        loss_r = log_cosh(*comp['r'])
        return loss_p + self.beta * (loss_t + loss_r)

    def _compute_sobolev(self, comp):
        # Sobolev = MSE(Value) + MSE(Gradient)
        def sobolev_term(p, t):
            val_loss = F.mse_loss(p, t)
            grad_p = p[:, 1:] - p[:, :-1]
            grad_t = t[:, 1:] - t[:, :-1]
            grad_loss = F.mse_loss(grad_p, grad_t)
            return self.alpha * val_loss + (1 - self.alpha) * grad_loss

        loss_p = sobolev_term(*comp['p'])
        loss_t = sobolev_term(*comp['t'])
        loss_r = sobolev_term(*comp['r'])
        return loss_p + self.beta * (loss_t + loss_r)

class WeightedPhysicsLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.1, beta=0.1, time_penalty=1.0, normalizer=None, horizon=50, delta_value=30000):
        super(WeightedPhysicsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.time_penalty = time_penalty
        self.delta_value = delta_value

        # Gear Efficiency
        self.efficiency = 0.97
        
        if normalizer is not None:
            # MinMaxScaler의 속성을 사용하여 버퍼 등록
            # Torque
            self.register_buffer('t_max_abs', torch.tensor(normalizer.posi_nega_scaler.scale_[-2], dtype=torch.float32))
            
            # RPM
            self.register_buffer('r_max_abs', torch.tensor(normalizer.posi_nega_scaler.scale_[-1], dtype=torch.float32))
            
            # Target Power (Target Scaler index 0)
            self.register_buffer('p_max_abs', torch.tensor(normalizer.power_scaler.scale_[0], dtype=torch.float32))

        # Set time weight (EXP)
        # t_steps = torch.linspace(0, self.time_penalty, steps=horizon)
        # val_w = torch.exp(t_steps)
        # grad_w = torch.exp(t_steps)

        # # Set time weight
        val_w = torch.linspace(1.0, 1.0 + self.time_penalty, steps=horizon)
        grad_w = torch.linspace(1.0, 1.0 + self.time_penalty, steps=horizon)

        # Gradient weight는 차분 데이터(SEQ_LEN-1)에 맞게 조정
        self.register_buffer('val_w', val_w)
        self.register_buffer('grad_w', grad_w[:-1]) 

    def forward(self, pred, target, input_tensor):
        # Weights normalization
        norm_val_w = self.val_w * (self.val_w.size(0) / self.val_w.sum())
        norm_grad_w = self.grad_w * (self.grad_w.size(0) / self.grad_w.sum())

        # input_tensor: [Batch, Seq, Features]
        # Take torque and RPM
        torque_norm = pred[:, :, -2]
        rpm_norm = pred[:, :, -1]
        
        # MaxAbsScaler Reverse Transform : (Norm * Range)
        torque = torque_norm * self.t_max_abs
        rpm = rpm_norm * self.r_max_abs

        # Data Loss
        # if "pred, target" shape is [Batch, Seq, 2], make into [Batch, Seq(power)]
        pw_pred = pred[:, :, 0]
        pw_target = target[:, :, 0]

        mse_element = (pw_pred - pw_target) ** 2
        weighted_mse = (mse_element * norm_val_w).mean()

        delta_pred = pw_pred[:, 1:] - pw_pred[:, :-1]
        delta_target = pw_target[:, 1:] - pw_target[:, :-1]
        grad_mse_element = (delta_pred - delta_target) ** 2
        weighted_grad_mse = (grad_mse_element * norm_grad_w).mean()


        ################## Physics Loss (W) ##################
        # Power = Torque * (2 * pi * RPM / 60)
        omega = (2 * torch.pi * rpm) / 60.0
        p_mech = torque * omega
        
        # Regenerative braking
        p_prop = torch.where(
            torque >= 0,
            p_mech / self.efficiency,
            p_mech * self.efficiency  # Regenerative braking
        )
        
        # Total Power (P_prop + P_aux)
        p_phys_real = p_prop
        
        # Normalize again
        # p_phys_norm = Real / Range
        p_phys_norm = p_phys_real / self.p_max_abs
        # physics_loss = torch.mean((pw_pred - p_phys_norm) ** 2)
        delta = self.delta_value / self.p_max_abs
        physics_loss = F.huber_loss(pw_pred, p_phys_norm, delta=delta)
        ################## Physics Loss (W) END ##################


        ################## Torque loss ##################
        torque_pred = pred[:, :, -2]
        torque_target = target[:, :, -2]

        torque_mse_element = (torque_pred - torque_target) ** 2
        # torque_mse = torque_mse_element.mean()
        weighted_torque_mse = (torque_mse_element * norm_val_w).mean()

        # Torque gradient loss
        delta_torque_pred = torque_pred[:, 1:] - torque_pred[:, :-1]
        delta_torque_target = torque_target[:, 1:] - torque_target[:, :-1]

        torque_grad_mse_element = (delta_torque_pred - delta_torque_target) ** 2
        # torque_grad_mse = torque_grad_mse_element.mean()
        weighted_grad_torque_mse = (torque_grad_mse_element * norm_grad_w).mean()

        # total_torque_loss = (self.alpha * torque_mse) + ((1 - self.alpha) * torque_grad_mse)
        total_torque_loss = (self.alpha * weighted_torque_mse) + ((1 - self.alpha) * weighted_grad_torque_mse)
        ################## Torque loss END ##################


        ################## RPM loss ##################
        rpm_pred = pred[:, :, -1]
        rpm_target = target[:, :, -1]

        rpm_mse_element = (rpm_pred - rpm_target) ** 2
        # rpm_mse = rpm_mse_element.mean()
        weighted_rpm_mse = (rpm_mse_element * norm_val_w).mean()

        # RPM gradient loss
        delta_rpm_pred = rpm_pred[:, 1:] - rpm_pred[:, :-1]
        delta_rpm_target = rpm_target[:, 1:] - rpm_target[:, :-1]

        rpm_grad_mse_element = (delta_rpm_pred - delta_rpm_target) ** 2
        # rpm_grad_mse = rpm_grad_mse_element.mean()
        weighted_grad_rpm_mse = (rpm_grad_mse_element * norm_grad_w).mean()

        # total_rpm_loss = (self.alpha * rpm_mse) + ((1 - self.alpha) * rpm_grad_mse)
        total_rpm_loss = (self.alpha * weighted_rpm_mse) + ((1 - self.alpha) * weighted_grad_rpm_mse)
        ################## RPM loss END ##################


        # Final loss
        data_loss = (self.alpha * weighted_mse) + ((1 - self.alpha) * weighted_grad_mse)
        final_loss = data_loss + (self.gamma * physics_loss) + (self.beta * total_torque_loss) + (self.beta * total_rpm_loss)

        raw_mse = mse_element.mean()

        loss_dict = {
            "total_loss": final_loss,
            "weighted_data_loss": data_loss,
            "weighted_physics_loss": (self.gamma * physics_loss),
            "weighted_torque_loss": (self.beta * total_torque_loss),
            "weighted_rpm_loss": (self.beta * total_rpm_loss),
            "raw_mse": raw_mse
        }
        return final_loss, raw_mse, loss_dict
    
    def update_weights(self, new_time_penalty):
        self.time_penalty = new_time_penalty
        # 새로운 가중치 계산 (기존 장치 유지)
        new_val_w = torch.linspace(1.0, 1.0 + self.time_penalty, steps=len(self.val_w)).to(self.val_w.device)
        new_grad_w = torch.linspace(1.0, 1.0 + self.time_penalty, steps=len(self.grad_w) + 1).to(self.grad_w.device)
        
        self.val_w.copy_(new_val_w)
        self.grad_w.copy_(new_grad_w[:-1])

class WeightedPhysicsLoss_ver2(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.1, beta=0.1, time_penalty=1.0, normalizer=None, horizon=50):
        super(WeightedPhysicsLoss_ver2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.time_penalty = time_penalty

        # Gear Efficiency
        self.efficiency = 0.97
        
        if normalizer is not None:
            # MinMaxScaler의 속성을 사용하여 버퍼 등록
            # Torque
            self.register_buffer('t_max_abs', torch.tensor(normalizer.posi_nega_scaler.scale_[-2], dtype=torch.float32))
            
            # RPM
            self.register_buffer('r_max_abs', torch.tensor(normalizer.posi_nega_scaler.scale_[-1], dtype=torch.float32))
            
            # Target Power (Target Scaler index 0)
            self.register_buffer('p_max_abs', torch.tensor(normalizer.power_scaler.scale_[0], dtype=torch.float32))

        # Set time weight (EXP)
        # t_steps = torch.linspace(0, self.time_penalty, steps=horizon)
        # val_w = torch.exp(t_steps)
        # grad_w = torch.exp(t_steps)

        # # Set time weight
        val_w = torch.linspace(1.0, 1.0 + self.time_penalty, steps=horizon)
        grad_w = torch.linspace(1.0, 1.0 + self.time_penalty, steps=horizon)

        # Gradient weight는 차분 데이터(SEQ_LEN-1)에 맞게 조정
        self.register_buffer('val_w', val_w)
        self.register_buffer('grad_w', grad_w[:-1]) 

    def forward(self, pred, target, input_tensor):
        # Weights normalization
        norm_val_w = self.val_w * (self.val_w.size(0) / self.val_w.sum())
        norm_grad_w = self.grad_w * (self.grad_w.size(0) / self.grad_w.sum())

        # input_tensor: [Batch, Seq, Features]
        # Take torque and RPM
        torque_norm = pred[:, :, -2]
        rpm_norm = pred[:, :, -1]
        
        # MaxAbsScaler Reverse Transform : (Norm * Range)
        torque = torque_norm * self.t_max_abs
        rpm = rpm_norm * self.r_max_abs

        # Data Loss
        # if "pred, target" shape is [Batch, Seq, 2], make into [Batch, Seq(power)]
        pw_pred = pred[:, :, 0]
        pw_target = target[:, :, 0]

        mae_element = torch.abs(pw_pred - pw_target)
        weighted_mse = (mae_element * norm_val_w).mean()

        delta_pred = pw_pred[:, 1:] - pw_pred[:, :-1]
        delta_target = pw_target[:, 1:] - pw_target[:, :-1]
        grad_mse_element = (delta_pred - delta_target) ** 2
        weighted_grad_mse = (grad_mse_element * norm_grad_w).mean()


        ################## Physics Loss (W) ##################
        # Power = Torque * (2 * pi * RPM / 60)
        omega = (2 * torch.pi * rpm) / 60.0
        p_mech = torque * omega
        
        # Regenerative braking
        p_prop = torch.where(
            torque >= 0,
            p_mech / self.efficiency,
            p_mech * self.efficiency  # Regenerative braking
        )
        
        # Total Power (P_prop + P_aux)
        p_phys_real = p_prop
        
        # Normalize again
        # p_phys_norm = Real / Range
        p_phys_norm = p_phys_real / self.p_max_abs
        # physics_loss = torch.mean((pw_pred - p_phys_norm) ** 2)
        delta = 30000 / self.p_max_abs
        physics_loss = F.huber_loss(pw_pred, p_phys_norm, delta=delta)
        ################## Physics Loss (W) END ##################


        ################## Torque loss ##################
        torque_pred = pred[:, :, -2]
        torque_target = target[:, :, -2]

        torque_mse_element = (torque_pred - torque_target) ** 2
        # torque_mse = torque_mse_element.mean()
        weighted_torque_mse = (torque_mse_element * norm_val_w).mean()

        # Torque gradient loss
        delta_torque_pred = torque_pred[:, 1:] - torque_pred[:, :-1]
        delta_torque_target = torque_target[:, 1:] - torque_target[:, :-1]

        torque_grad_mse_element = (delta_torque_pred - delta_torque_target) ** 2
        # torque_grad_mse = torque_grad_mse_element.mean()
        weighted_grad_torque_mse = (torque_grad_mse_element * norm_grad_w).mean()

        # total_torque_loss = (self.alpha * torque_mse) + ((1 - self.alpha) * torque_grad_mse)
        total_torque_loss = (self.alpha * weighted_torque_mse) + ((1 - self.alpha) * weighted_grad_torque_mse)
        ################## Torque loss END ##################


        ################## RPM loss ##################
        rpm_pred = pred[:, :, -1]
        rpm_target = target[:, :, -1]

        rpm_mse_element = (rpm_pred - rpm_target) ** 2
        # rpm_mse = rpm_mse_element.mean()
        weighted_rpm_mse = (rpm_mse_element * norm_val_w).mean()

        # RPM gradient loss
        delta_rpm_pred = rpm_pred[:, 1:] - rpm_pred[:, :-1]
        delta_rpm_target = rpm_target[:, 1:] - rpm_target[:, :-1]

        rpm_grad_mse_element = (delta_rpm_pred - delta_rpm_target) ** 2
        # rpm_grad_mse = rpm_grad_mse_element.mean()
        weighted_grad_rpm_mse = (rpm_grad_mse_element * norm_grad_w).mean()

        # total_rpm_loss = (self.alpha * rpm_mse) + ((1 - self.alpha) * rpm_grad_mse)
        total_rpm_loss = (self.alpha * weighted_rpm_mse) + ((1 - self.alpha) * weighted_grad_rpm_mse)
        ################## RPM loss END ##################


        # Final loss
        data_loss = (self.alpha * weighted_mse) + ((1 - self.alpha) * weighted_grad_mse)
        final_loss = data_loss + (self.gamma * physics_loss) + (self.beta * total_torque_loss) + (self.beta * total_rpm_loss)

        raw_mse = mae_element.mean()

        loss_dict = {
            "total_loss": final_loss,
            "weighted_data_loss": data_loss,
            "weighted_physics_loss": (self.gamma * physics_loss),
            "weighted_torque_loss": (self.beta * total_torque_loss),
            "weighted_rpm_loss": (self.beta * total_rpm_loss),
            "raw_mse": raw_mse
        }
        return final_loss, raw_mse, loss_dict