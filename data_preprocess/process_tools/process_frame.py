import os
import cv2
import rosbag
from cv_bridge import CvBridge
from tqdm import tqdm
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