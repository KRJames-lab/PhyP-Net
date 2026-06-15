import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

DROPOUT = 0.2
MULTIHEAD_H_DIM = 64

class MultiHeadPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_vars=2, dropout=DROPOUT):
        super().__init__()
        self.num_vars = num_vars

        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # Head 1 : 0-1s
        self.head_short = nn.Sequential(
            nn.Linear(hidden_dim, 10 * num_vars)
        )

        # Head 2 : 1-3s (20 steps)
        self.head_mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 20 * num_vars)
        )
        
        # Head 3 : 3-5s (20 steps)
        self.head_long = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 20 * num_vars)
        )

    def forward(self, x):
        shared = self.shared_layer(x)

        out1 = self.head_short(shared)
        out2 = self.head_mid(shared)
        out3 = self.head_long(shared)

        return torch.cat([out1, out2, out3], dim=-1)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, out):
        attn_scores = self.attn_layer(out)
        attn_weights = F.softmax(attn_scores / self.temperature, dim=1)

        context_vector = torch.bmm(attn_weights.transpose(1, 2), out)
        return context_vector.squeeze(1), attn_weights

class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.project = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        identity = self.project(x)

        out, _ = self.lstm(x)
        out = self.dropout(out)

        out = self.norm(out + identity)
        # print(out.shape)
        return out

class DeepResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualLSTMBlock(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_vars=2, temperature=0.1):
        """
        [Args]
        input_size : Num of input features (Speed, Acceleration, Gradient)
        hidden_size : Number of LSTM node (64, 128)
        num_layers : Number of LSTM layers (1~3)
        output_size : Num of output features (Power)
        temperature : Attention pooling temperature
        """
        super(LSTMModel, self).__init__()
        self.num_vars = num_vars
        self.output_size = output_size
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers

        # self.lstm = nn.LSTM(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout = 0.2
        # )
        self.encoder = DeepResidualLSTM(input_size, hidden_size, num_layers)
        self.attention = AttentionPooling(hidden_size, temperature=temperature)
        self.multihead = MultiHeadPredictor(input_dim=hidden_size * 2, hidden_dim=MULTIHEAD_H_DIM, num_vars=num_vars)
        self.shortcut = nn.Linear(hidden_size * 2, output_size * num_vars)

        nn.init.xavier_uniform_(self.shortcut.weight)
        with torch.no_grad():
            self.shortcut.weight.mul_(0.1) 
    
        nn.init.zeros_(self.shortcut.bias)

    def forward(self, x):
        """
        [Input]
        x: (Batch_Size, Seq_Length, Input_Size) Tensor
        """

        # LSTM propagation
        # out shape : (Batch_Size, Seq_Length, Hidden_Size)
        # out, _ = self.lstm(x)

        # Residual LSTM propagation
        out = self.encoder(x)

        # Attention Pooling
        attn_out, weights = self.attention(out)

        # Extract last info
        last_out = out[:, -1, :]

        # Combination
        combined = torch.cat([attn_out, last_out], dim=-1)

        raw_prediction = self.multihead(combined) + self.shortcut(combined)

        # (Batch, 100) -> (Batch, 50, 2)
        prediction = raw_prediction.view(-1, self.output_size, self.num_vars)

        power_pred = torch.tanh(prediction[:, :, 0])    # [-1, 1]
        torque_pred = torch.tanh(prediction[:, :, 1])
        rpm_pred = torch.tanh(prediction[:, :, 2])

        # Stacking
        final_prediction = torch.stack([power_pred, torque_pred, rpm_pred], dim=-1)

        return final_prediction, weights

class BasicLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_vars):
        """
        [Args]
        input_size : 입력 피처의 개수 (Speed, Acceleration, Gradient 등)
        hidden_size : LSTM 은닉 노드의 개수
        num_layers : LSTM 레이어의 층수
        output_size : 예측하고자 하는 미래 타임 스텝의 길이 (여기서는 50)
        num_vars : 예측할 변수의 개수 (여기서는 3)
        """
        super(BasicLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size # 예측 타임 스텝 (50)
        self.num_vars = num_vars       # 변수 개수 (3)

        # LSTM 레이어 정의
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # 레이어가 1개일 때는 드롭아웃이 의미가 없습니다.
        )
        self.predictor = MultiHeadPredictor(
            input_dim=hidden_size, 
            hidden_dim=64, 
            num_vars=num_vars
        )
        
        # 선형 레이어(Linear Layer): 은닉 상태를 (output_size * num_vars) 크기로 변환
        # 3개 변수 * 50개 스텝 = 150개의 출력 값을 생성합니다.
        # self.fc = nn.Linear(hidden_size, self.output_size * self.num_vars)

    def forward(self, x):
        """
        [Input]
        x: (Batch_Size, Seq_Length, Input_Size) 형태의 텐서(Tensor)
        """
        # 은닉 상태(Hidden State)와 셀 상태(Cell State) 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 순전파(Propagation)
        # out shape: (Batch_Size, Seq_Length, Hidden_Size)
        out, hidden = self.lstm(x, (h0, c0))

        # 다대일(Many-to-One) 방식: 마지막 타임 스텝의 출력값만 사용
        # last_out shape: (Batch_Size, Hidden_Size)
        last_out = out[:, -1, :]

        # 최종 예측값 생성
        # prediction shape: (Batch_Size, output_size * num_vars)
        prediction = self.predictor(last_out)

        # (Batch_Size, output_size, num_vars) 형태로 형태 변환(Reshape)
        # 즉, (Batch_Size, 50, 3)의 형태로 출력됩니다.
        prediction = prediction.view(-1, self.output_size, self.num_vars)

        return prediction, hidden

class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_vars=11, out_vars=3):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_vars = in_vars
        self.out_vars = out_vars

        # 1. Channel-Independent Linear Layers
        self.Linear = nn.Conv1d(
            in_channels=self.in_vars, 
            out_channels=self.pred_len * self.in_vars, 
            kernel_size=self.seq_len, 
            groups=self.in_vars
        )
        
        # 2. Variable Projection Layer
        # NLinear로 예측된 11개의 미래 시계열을 바탕으로 최종 3개 변수를 도출
        # 이 단계에서 Power, Torque, RPM 간의 관계가 학습됨
        self.projection = nn.Linear(self.in_vars, self.out_vars)

    def forward(self, x):
        # x: [Batch, Seq_Len, In_Vars]

        current_batch_size = x.shape[0]

        # [Step 1] Normalization (정규화)
        seq_last = x[:, -1:, :].detach() # [Batch, 1, 11]
        x = x - seq_last

        # [Step 2] Temporal Forecasting (시간 축 예측 - 채널 독립)
        # 결과 저장용 텐서: [Batch, Pred_Len, In_Vars]
        x = x.transpose(1, 2)
        forecast = self.Linear(x)

        # 3) Shape 복원: [B, In_Vars, Pred_Len]으로 먼저 만든 후 축 변경
        # [B, 11 * Pred_Len, 1] -> [B, 11, Pred_Len]
        forecast = forecast.view(current_batch_size, self.in_vars, self.pred_len)
        # [B, 11, Pred_Len] -> [B, Pred_Len, 11]
        forecast = forecast.transpose(1, 2)

        # [Step 3] Variable Projection (변수 축 투영)
        # [Batch, Pred_Len, 11] -> [Batch, Pred_Len, 3]
        output = self.projection(forecast)

        # [Step 4] Denormalization (역정규화)
        # 타겟이 되는 3개 변수(Power 등)의 마지막 값을 더해줌
        target_indices = [9, 7, 8]
        target_last = seq_last[:, :, target_indices] # [Batch, 1, 3]

        output = output + target_last
        
        return output, output

class PatchTST(nn.Module):
    def __name__(self):
        print("PatchTST")
        return "PatchTST" 

    def __init__(self, seq_len, pred_len, in_vars=11, out_vars=3, patch_len=16, stride=8, d_model=32, n_heads=4, n_layers=3, dropout=0.1):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.in_vars = in_vars
        
        # 1. 패치 개수 계산 및 패딩 크기 결정
        # ((seq_len - patch_len) // stride) + 1 로직 보강
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        self.re_padding = (self.num_patch - 1) * stride + patch_len - seq_len
        
        # 2. Embedding & Positional Encoding
        self.input_layer = nn.Linear(patch_len, d_model)
        self.W_pos = nn.Parameter(torch.randn(1, self.num_patch, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            batch_first=True, dropout=dropout, activation='gelu' # 논문은 주로 GELU 사용
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Prediction Head (Flatten + Linear)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_predict = nn.Linear(d_model * self.num_patch, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, In_Vars]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # [Step 1] RevIN 스타일 정규화 (평균/분산 활용)
        # 각 채널별 평균과 표준편차 계산
        mean = torch.mean(x, dim=1, keepdim=True).detach() # [B, 1, V]
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # [B, 1, V]
        x = (x - mean) / std
        
        # [Step 2] Padding (unfold 시 데이터 손실 방지)
        if self.re_padding > 0:
            last_value = x[:, -1:, :]
            padding = last_value.repeat(1, self.re_padding, 1)
            x = torch.cat([x, padding], dim=1)
        
        # [Step 3] Channel Independence & Patching
        x = x.transpose(1, 2) 
        batch_size, n_vars, seq_len = x.shape
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patches.append(x[:, :, i:i+self.patch_len])
        x = torch.stack(patches, dim=2) # [Batch, In_Vars, P_Num, P_Len]
        x = x.reshape(batch_size * self.in_vars, self.num_patch, self.patch_len) # [B*V, P_Num, P_Len]
        
        # [Step 4] Embedding & Transformer
        x = self.input_layer(x) + self.W_pos
        x = self.transformer_encoder(x) # [B*V, P_Num, d_model]
        
        # [Step 5] Head & Reshape
        x = self.flatten(x)
        x = self.linear_predict(x) # [B*V, Pred_Len]
        x = x.reshape(batch_size, self.in_vars, -1).transpose(1, 2) # [B, Pred_Len, 11]
        
        # [Step 6] 역정규화 (Denormalization)
        # 11개 전체를 역정규화한 뒤 필요한 타겟(9, 7, 8)만 추출
        x = x * std + mean
        
        # 최종적으로 Power(9), Torque(7), RPM(8) 순으로 반환
        target_indices = [9, 7, 8]
        out = x[:, :, target_indices]
        
        return out, out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # chomp_size가 0일 경우 불필요한 슬라이싱 방지
        if self.chomp_size == 0:
            return x
        # contiguous()는 메모리 복사를 일으키므로 꼭 필요한 경우에만 사용하거나
        # 모델의 마지막 단계에서만 처리하는 것이 유리할 수 있습니다.
        return x[:, :, :-self.chomp_size].contiguous() 

class ChausalConv1d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, groups=1):
        super(ChausalConv1d, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, 
                              dilation=dilation, groups=groups)
        
        # 2. BatchNorm1d 추가 (채널 수인 n_outputs를 인자로 받음)
        self.bn = nn.BatchNorm1d(n_outputs)

        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)

    def forward(self, x):
        return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout, groups):
        super(TemporalBlock, self).__init__()
        # 모든 컨볼루션 레이어에 groups를 적용합니다.
        self.conv1 = ChausalConv1d(n_inputs, n_outputs, kernel_size, stride, dilation, dropout, groups=groups)
        self.conv2 = ChausalConv1d(n_outputs, n_outputs, kernel_size, stride, dilation, dropout, groups=groups)
        
        
        if n_inputs != n_outputs:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_inputs, n_outputs, 1, groups=groups),
                nn.BatchNorm1d(n_outputs)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, output_size, num_vars, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # input_size(11)를 groups로 설정하여 채널 독립성 확보
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                     dilation=dilation_size, dropout=dropout, groups=1)]

        self.tcn = nn.Sequential(*layers)
        self.predictor = MultiHeadPredictor(input_dim=num_channels[-1], hidden_dim=64, num_vars=num_vars)
        self.output_size = output_size
        self.num_vars = num_vars

    def forward(self, x):
        # x: (Batch, Seq_len, Features) -> (512, 150, 11)
        # x = x.permute(0, 2, 1) # (Batch, Features, Seq_len) -> (512, 11, 150)
        y = self.tcn(x)
        last_step = y[:, :, -1]
        prediction = self.predictor(last_step)
        prediction = prediction.view(-1, self.output_size, self.num_vars)
        return prediction, prediction


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        [Args]
        input_size : Num of input features (Speed, Acceleration, Gradient)
        hidden_size : Number of GRU node (64, 128)
        num_layers : Number of GRU layers (1~3)
        output_size : Num of output features (Power)
        """
        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout = 0.2
        )

        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        [Input]
        x: (Batch_Size, Seq_Length, Input_Size) Tensor
        """

        # Initialize "Hidden State", No "Cell state" in GRU
        # x.size(0) : Batch_Size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU propagation
        # out shape : (Batch_Size, Seq_Length, Hidden_Size)
        out, _ = self.gru(x, h0)

        # Many-to-One Vector
        # Use only "Last Time Step" for predicting future
        last_out = out[:, -1, :]    # Every batch's, last time step, every features

        prediction = self.fc(last_out)

        return prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 위치 정보를 담을 행렬 생성 (Seq_Len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 사인(sin), 코사인(cos) 함수를 교차하여 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (Batch, Seq, Feature) 형태에 맞추기 위해 차원 추가 -> (1, Seq, Feature)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        # 입력 시퀀스 길이에 맞춰서 잘라서 더해줌
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        """
        [Args]
        input_size : 입력 피처 개수 (예: 5)
        d_model    : 모델 내부 차원 (예: 64 or 128) - 반드시 nhead로 나누어 떨어져야 함!
        nhead      : 멀티헤드 어텐션 헤드 개수 (예: 4 or 8)
        num_layers : 인코더 레이어 층 수 (예: 2 or 3)
        output_size: 출력 개수 (예: 1)
        """
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        self.d_model = d_model

        # 1. Input Projection (Feature 9 -> d_model 64)
        self.input_linear = nn.Linear(input_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. Final Decoder (Regression Head)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, src):
        # src: (Batch_Size, Seq_Len, Input_Size)
        
        # 1. Embedding & Scaling
        src = self.input_linear(src) * math.sqrt(self.d_model)
        
        # 2. Add Positional Encoding
        src = self.pos_encoder(src)
        
        # 3. Encoder Pass
        # output: (Batch, Seq, d_model)
        output = self.transformer_encoder(src)
        
        # 4. Use Last Token Only (Many-to-One)
        # Take last vector and predict number of "OUTPUT_SIZE"
        last_output = output[:, -1, :]
        
        # 5. Prediction
        prediction = self.fc(last_output)
        
        return prediction

# Test code
if __name__ == "__main__":
    INPUT_SIZE = 3   
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1 
    
    # Model generate
    model = LSTMAttentionModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    print("Check model's structure:")
    print(model)
    
    dummy_input = torch.randn(8, 10, 3)
    dummy_output = model(dummy_input)
    
    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", dummy_output.shape)