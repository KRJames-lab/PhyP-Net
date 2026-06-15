from pathlib import Path
import numpy as np

def padding(*args):
    processed_data = []
    length = 600
    
    for data in args:
        current_len = len(data)
        if current_len >= length:
            new_data = data[:length]
        elif current_len < length:
            pad_width = length - current_len
            new_data = np.pad(data, (0, pad_width), mode='edge')

        processed_data.append(new_data)
    
    return tuple(processed_data)

class ReshapeData:
    def __init__(self, path: str):
        self.path = path
    
    def reshapedata(self):
        path = Path(self.path)
        dataset_list = []

        for time_path in path.rglob('time.npy'):
            parent_dir = time_path.parent

            try:
                time = np.load(time_path)
                speed = np.load(parent_dir / "speed.npy")
                acceleration = np.load(parent_dir / "acceleration.npy")
                gradient = np.load(parent_dir / "gradient.npy")
                power = np.load(parent_dir / "power.npy")

                time, speed, acceleration, gradient, power = padding(time, speed, acceleration, gradient, power)

                single_sample = np.column_stack([time, speed, acceleration, gradient, power])

                dataset_list.append(single_sample)
            
            except FileNotFoundError as e:
                print("No data")
                continue

        if dataset_list:
            final_3d_array = np.array(dataset_list)
            return final_3d_array

if __name__ == "__main__":
    PATH = "/root/comma2k19_10Hz"

    rd = ReshapeData(PATH)
    
    reshape3d = rd.reshapedata