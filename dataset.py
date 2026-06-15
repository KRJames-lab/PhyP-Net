import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DrivingDataset:
    def __init__(self, segments, seq_len, horizon, input_size):
        """
        segments: (N_segments, Time_steps, Features)
        seq_len: Window Size (Paste data's size)
        """
        self.segments = [torch.tensor(s, dtype=torch.float32) for s in segments]
        self.seq_len = seq_len
        self.horizon = horizon
        self.input_size = input_size
        
        self.indices = []
        for seg_idx, segment in enumerate(self.segments):
            num_windows = len(segment) - seq_len - horizon + 1
            for start_idx in range(num_windows):
                self.indices.append((seg_idx, start_idx))

        # X : (Number of window, Time_steps, Feature)
        # self.X, self.y, self.y_input = self._create_windows(segments)

        # self.X = torch.tensor(self.X, dtype=torch.float32)
        # self.y = torch.tensor(self.y, dtype=torch.float32)
        # self.y_input = torch.tensor(self.y_input, dtype=torch.float32)

    def _create_windows(self, segments):
        all_xs = []
        all_ys = []
        all_y_inputs = []
        
        for segment in segments:
            seg_x = segment[:, 0:self.input_size]
            seg_y = segment[:, self.input_size:]
            
            for t in range(len(segment) - self.seq_len - self.horizon + 1):
                x_window = seg_x[t : t+self.seq_len]
                y_target = seg_y[t+self.seq_len : t+self.seq_len+self.horizon]
                y_input_window = seg_x[t + self.seq_len : t + self.seq_len + self.horizon]

                all_xs.append(x_window)
                all_ys.append(y_target)
                all_y_inputs.append(y_input_window)
                
        return np.array(all_xs), np.array(all_ys), np.array(all_y_inputs)

    # len(dataset) : Return input's length
    def __len__(self):
        return len(self.indices)

    # dataset[n]
    def __getitem__(self, idx):
        # 1. 인덱스를 통해 해당 세그먼트와 시작 위치를 찾음
        seg_idx, start_idx = self.indices[idx]
        segment = self.segments[seg_idx]
        
        # 2. 필요한 부분만 실시간으로 슬라이싱 (메모리 복사 최소화)
        # 과거 데이터 (300스텝)
        x_window = segment[start_idx : start_idx + self.seq_len, :self.input_size]
        
        # 미래 정답 Power (50스텝)
        y_target = segment[start_idx + self.seq_len : start_idx + self.seq_len + self.horizon, self.input_size:]

        # 미래 입력 Features (50스텝)
        y_input_window = segment[start_idx + self.seq_len : start_idx + self.seq_len + self.horizon, :self.input_size:]
        
        return x_window, y_target, y_input_window