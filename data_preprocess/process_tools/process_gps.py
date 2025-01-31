import pynmea2
import numpy as np
import os
from process_tools.helpers import convert_time_to_rostime,devide_name_to_time

gps_file_name=[
'20200421_170039-sunset1_concat',
'20200422_172431-sunset2_concat',
'20200424_151015-daytime_concat',
'20200428_091154-morning_concat',
'20200429_061912-sunrise_concat'
]

file_name=[
'ss1',
'ss2',
'dt',
'mn',
'sr']

bias = [
1587452582.35,
1587540271.65,
1587705130.80,
1588029265.73,
1588105232.91]

# 函数:get_gps
# 用途:输入nmea文件路径，输出np格式的经度纬度时间戳
def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding='utf-8')

    latitudes, longitudes, timestamps = [], [], []

    first_timestamp = None
    previous_lat, previous_lon = 0, 0
    output_file = open('nmea_time.txt', 'w')
    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if first_timestamp is None:
                first_timestamp = msg.timestamp
            if msg.sentence_type not in ['GSV', 'VTG', 'GSA']:
                output_file.write("time:"+f"hour{msg.timestamp.hour} minute{msg.timestamp.minute} second{msg.timestamp.second}"+'\n')
                #print(msg.timestamp, msg.latitude, msg.longitude)
                #print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude); longitudes.append(msg.longitude); timestamps.append(timestamp_diff)
                    previous_lat, previous_lon = msg.latitude, msg.longitude

        except pynmea2.ParseError as e:
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps))).T

# 函数:process_gps
# 用途:输入nmea文件名，将读取到的gps信息写进txt存储到对应文件夹
def process_gps(gps_file_name, file_name, bias):
    """
    Process GPS data from an NMEA file and write the processed data to a txt file.

    Parameters:
    - gps_file_name: str - The name of the GPS NMEA file.
    - file_name: str - The name of the output file.
    - bias: float - The bias value to be added to the GPS data.
    """
    # 定义GPS数据文件的路径
    file_path = f'/root/autodl-fs/Brizbane_dataset/{gps_file_name}.nmea'
    
    # 确保文件存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GPS file not found: {file_path}")
    
    # 读取GPS数据
    gps_data = get_gps(file_path)
    
    # 调整数据
    gps_data[:, 2] = gps_data[:, 2].astype(float) + float(bias)  # sr
    
    # 确保输出目录存在
    output_dir = f'/root/autodl-tmp/processed_data/{file_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 将gps_data写入txt文件
    with open(f'{output_dir}/gps.txt', 'w') as f:
        for row in gps_data:
            f.write(' '.join(map(str, row)) + '\n')
    
    
    # 返回处理后的GPS数据
    return gps_data

def process_all_gps():
    for index in range(len(file_name)):
        process_gps(gps_file_name[index],file_name[index],bias[index])