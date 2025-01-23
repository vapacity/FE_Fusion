import os
import cv2
import rosbag
from cv_bridge import CvBridge
from tqdm import tqdm
file_name=[
# 'dvs_vpr_2020-04-21-17-03-03',
# 'dvs_vpr_2020-04-22-17-24-21',
'dvs_vpr_2020-04-24-15-12-03',
# 'dvs_vpr_2020-04-27-18-13-29',
'dvs_vpr_2020-04-28-09-14-11',
'dvs_vpr_2020-04-29-06-20-23'
]
# 函数: process_frame
# 用途：从bag_file中读取图像帧信息，以frame_initerval为读取的两帧之间的间隔
def process_frame(bag_file, output_dir, frame_interval):
    bridge = CvBridge()
    #max_output = 10
    #output_count = 0
    last_time = None
    print(output_dir)
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    with rosbag.Bag(bag_file, 'r') as bag:
        # 获取消息总数，用于进度条
        total_messages = bag.get_message_count(topic_filters=['/dvs/image_raw'])
        pbar = tqdm(total=total_messages,desc='processing frame')
        for topic, msg, t in bag.read_messages(topics=['/dvs/image_raw']):
            time = t.to_sec()
            if last_time is None or (time - last_time) >= frame_interval:
                last_time = time
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                
                # 构造输出文件路径
                image_filename = os.path.join(output_dir, f'{time}.png')
                
                # 保存图像到文件
                cv2.imwrite(image_filename, cv_image)
                pbar.update(1)  # 更新进度条

                # 如果输出数量达到最大限制，则停止
                #if output_count >= max_output:
                #    break
        pbar.close()  # 关闭进度条

# 调用函数
for name in file_name:
    bag_file = '/root/autodl-fs/Brizbane_dataset/'+name+'.bag'
    output_dir = '/root/autodl-tmp/processed_data/'+name+'/frame/'
    process_frame(bag_file, output_dir, frame_interval=0.25)  # 这里设定了frame_interval，根据需要调整
