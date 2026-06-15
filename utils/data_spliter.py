import os
import numpy as np
import glob
import pandas as pd
from pathlib import Path
import random

from processor import DataNormalizer

TRAIN = 0.7
VALID = 0.2
TEST = 0.1

DATA_PATH = "/root/tesla-model3_data"
SAVE_PATH = "segmented_data"

data_path = Path(DATA_PATH)

all_drive_dict = {}

def load_driving_data(path: Path):
    for i, file in enumerate(sorted(path.glob("*.csv"))):
        print(f"\n=== Drive{i} now processing ===")

        driving_data = pd.read_csv(file, low_memory=False)

        exclude_list = ['Idle|0', 'Charging|1', 'Standby|2']
        condition = driving_data['veh_state_drive'].isin(exclude_list)
        target_indices = driving_data[condition].index.tolist()

        print(f"Total data : {len(driving_data['veh_state_drive'])}")
        print(f"Non drive data : {len(target_indices)}")
        print(f"Number of data remaining driving data : {len(driving_data['veh_state_drive']) - len(target_indices)}")

        clean_driving_data = driving_data[~driving_data['veh_state_drive'].isin(exclude_list)]

        # input
        time = clean_driving_data['Time (abs)']
        speed = clean_driving_data['veh_speed (kph)']
        acceleration = clean_driving_data['RCM_longitudinalAccel (m/s^2)']
        elevation = clean_driving_data['veh_elevation (M)']
        gradient = np.diff(elevation, prepend=elevation.iloc[0])
        pedal = clean_driving_data['pedal_accel (per)']
        delta_pedal = np.diff(pedal, prepend=pedal.iloc[0])

        speed_squr = speed ** 2
        jerk = np.diff(acceleration, prepend=acceleration.iloc[0])
        speed_accel = speed * acceleration

        dif_torque = clean_driving_data['DIF_torqueCommand (Nm)']
        dir_torque = clean_driving_data['DIR_torqueCommand (Nm)']
        total_torque = dif_torque + dir_torque

        dif_axle = clean_driving_data['DIF_axleSpeed (rpm)']
        dir_axle = clean_driving_data['DIR_axleSpeed (rpm)']
        total_axle = (dif_axle + dir_axle) / 2

        bms_volt = clean_driving_data['BMS_packVoltage (V)']

        # output
        dif_power = clean_driving_data['DIF_elecPower (kW)']
        dir_power = clean_driving_data['DIR_elecPower (kW)']
        total_power = dif_power + dir_power
        delta_power = np.diff(total_power, prepend=total_power.iloc[0])

        np_driving_data = np.column_stack((
            time, 

            speed_squr,
            pedal,

            speed, 
            acceleration,
            jerk, 
            gradient,
            delta_pedal, 
            total_torque, 
            total_axle,
            total_power * 1000,  # Change to W (kW -> W)
            delta_power * 1000,  # Change to W (kW -> W)

            total_power * 1000,  # Change to W (kW -> W)
            total_torque, 
            total_axle
            ))

        print(f"Drive{i} has done")

        all_drive_dict[i] = np_driving_data
    
    return all_drive_dict

def split_data(drive_dict: dict, TRAIN=0.7, VALID=0.2, TEST=0.1):
    print("\n=== Data split process start ===")
    
    keys = list(drive_dict.keys())
    values = list(drive_dict.values())
    
    drive_duration = [len(v) for v in values]
    np_drive_duration = np.array(drive_duration)
    
    sorted_drive_duration_idx = np.argsort(np_drive_duration)[::-1]

    total_drive_duration = sum(drive_duration)
    print(f"Total drive duration is : {total_drive_duration}")
    print(f"Ideal Train/Valid/Test length is : {total_drive_duration*TRAIN:.2f} / {total_drive_duration*VALID:.2f} / {total_drive_duration * TEST:.2f}")

    # '''
    # Take longer drives to Test set
    # '''
    # testset_dict = {}
    # testset_idx = []
    # testset_length = 0

    # for i in sorted_drive_duration_idx:
    #     real_key = keys[i]  

    #     if testset_length + len(drive_dict[real_key]) < total_drive_duration * TEST:
    #         testset_dict[real_key] = drive_dict[real_key]
    #         testset_length += len(testset_dict[real_key])
            
    #         testset_idx.append(real_key)

    # print(f"\nTest set driving length is : {testset_length}, {(testset_length/total_drive_duration*100):2f}%")
    # print(f"Test set indices (Keys) : {testset_idx}")

    '''
    Take longer drives to Test set
    '''
    testset_dict = {}
    testset_idx = [0, 1, 3, 12, 43, 45]
    testset_length = 0

    print(f"Test set indices (Keys) : {testset_idx}")

    for i, idx in enumerate(testset_idx):
        testset_dict[i] = drive_dict[idx]
        testset_length += len(testset_dict[i])

    print(f"\nTest set driving length is : {testset_length}, {(testset_length/total_drive_duration*100):2f}%")
    

    '''
    Split Train/Valid
    '''
    for key in testset_idx:
        drive_dict.pop(key, None)
    
    drive_dict_no_test = drive_dict

    # Shuffle drives
    shuffled_keys = list(drive_dict_no_test.keys())
    random.shuffle(shuffled_keys)

    # Make Train and Valid
    trainset_dict = {}
    validset_dict = {}
    trainset_length = 0
    validset_length = 0

    for i, key in enumerate(shuffled_keys):
        current_data = drive_dict_no_test[key]
        
        if validset_length + len(current_data) < total_drive_duration * VALID:
            validset_dict[i] = current_data
            validset_length += len(current_data)
        else:
            trainset_dict[i] = current_data
            trainset_length += len(current_data)
    
    print(f"\nTrain set driving length is : {trainset_length}, {(trainset_length/total_drive_duration*100):2f}%")
    print(f"Valid set driving length is : {validset_length}, {(validset_length/total_drive_duration*100):2f}%")
    print(f"Test set driving length is : {testset_length}, {(testset_length/total_drive_duration*100):2f}%")

    print(f"\nFinal checking Train + Valid + Test : {((trainset_length+validset_length+testset_length)/total_drive_duration*100):2f}%")
    
    return trainset_dict, validset_dict, testset_dict

def save_each_drive(drive_set, set_type: str):
    print(f"\n{set_type.upper()} is now saving...")

    for i, data in enumerate(drive_set.values()):
        np.save(os.path.join(SAVE_PATH, f"Drive_{set_type}", f"{set_type}_{i}"), data)

    print("Done")

def make_into_segments(drive_set, set_type: str):
    print(f"\nMake {set_type.upper()} segments is processing...")

    positive_path = os.path.join(SAVE_PATH, "positive_scaler.pkl")
    posi_nega_path = os.path.join(SAVE_PATH, "posi_nega_scaler.pkl")
    target_power_path = os.path.join(SAVE_PATH, "power_scaler.pkl")

    normalizer = DataNormalizer(feature_range=(0, 1))
    
    segmented_list = []
    for data in drive_set.values():
        segmented_list.append(data)

    if set_type == "train":
        print(f"Train is processing. Fitting scaler...")
        normalizer.fit(segmented_list)
        normalizer.save_scaler(positive_path, posi_nega_path, target_power_path)
    else:
        normalizer.load_scaler(positive_path, posi_nega_path, target_power_path)
    
    scaled_segments = normalizer.transform(segmented_list)

    np.save(os.path.join(SAVE_PATH, f"{set_type}_segments"), scaled_segments, allow_pickle=True)

    print("Making segments is done and scaler saved")
    print("Make sure the train set processd first")

if __name__ == "__main__":
    drive_dict = load_driving_data(data_path)

    trainset_dict, validset_dict, testset_dict = split_data(drive_dict)

    save_each_drive(trainset_dict, "train")
    save_each_drive(validset_dict, "valid")
    save_each_drive(testset_dict, "test")

    make_into_segments(trainset_dict, "train")
    make_into_segments(validset_dict, "valid")
    make_into_segments(testset_dict, "test")

    # *** 'split_data' test code *** #
    # dummy_input = {}
    # for i in range(1, 50):
    #     dummy_input[i] = np.random.rand(1000)

    # split_data(dummy_input)
    # *** 'split_data' test code end *** #