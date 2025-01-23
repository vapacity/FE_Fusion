import pynmea2
import numpy as np
from tqdm import tqdm
from helpers import read_gps_data,read_timestamp
gps_file_name=[
#'20200421_170039-sunset1_concat',
#'20200422_172431-sunset2_concat',
'20200424_151015-daytime_concat',
#'20200427_181204-night_concat',
'20200428_091154-morning_concat',
'20200429_061912-sunrise_concat'
]

file_name=[
'dvs_vpr_2020-04-21-17-03-03',
'dvs_vpr_2020-04-22-17-24-21',
#'dvs_vpr_2020-04-24-15-12-03',
'dvs_vpr_2020-04-27-18-13-29',
#'dvs_vpr_2020-04-28-09-14-11',
#'dvs_vpr_2020-04-29-06-20-23'
]



        
def interpolate_gps_data(timestamp_file, gps_file_path, interpolated_gps_path):
    gps_data = read_gps_data(gps_file_path)
    gps_timestamps = gps_data[:, 2]
    timestamps = read_timestamp(timestamp_file)
    interpolated_data = []

    gps_idx = 0
    
    for event_time in tqdm(timestamps, desc='Interpolating GPS data'):
        if event_time < gps_timestamps[0]:
            continue
        
        while gps_idx < len(gps_timestamps) - 1 and event_time > gps_timestamps[gps_idx + 1]:
            gps_idx += 1

        if gps_idx == 0 or gps_idx == len(gps_timestamps) - 1:
            # 无法在开始或结束时进行插值
            interpolated_lat, interpolated_lon = None, None
        else:
            # 进行线性插值
            t0, t1 = gps_timestamps[gps_idx], gps_timestamps[gps_idx + 1]
            lat0, lat1 = gps_data[gps_idx, 0], gps_data[gps_idx + 1, 0]
            lon0, lon1 = gps_data[gps_idx, 1], gps_data[gps_idx + 1, 1]
            alpha = (event_time - t0) / (t1 - t0)
            interpolated_lat = lat0 + alpha * (lat1 - lat0)
            interpolated_lon = lon0 + alpha * (lon1 - lon0)
        
        if interpolated_lat is not None and interpolated_lon is not None:
            interpolated_data.append([interpolated_lat, interpolated_lon, event_time])
    
    # 将插值后的数据保存到新文件中，保持高精度
    with open(interpolated_gps_path, 'w') as f:
        for data in interpolated_data:
            f.write(f"{repr(data[0])} {repr(data[1])} {repr(data[2])}\n")

for name in file_name:
    timestamp_file = '/root/autodl-tmp/processed_data/'+ name+'/timestamp.txt'
    gps_file =  '/root/autodl-tmp/processed_data/'+ name+'/gps.txt'
    interpolated_gps_file ='/root/autodl-tmp/processed_data/'+ name+'/interpolated_gps.txt'
    interpolate_gps_data(timestamp_file,gps_file,interpolated_gps_file)