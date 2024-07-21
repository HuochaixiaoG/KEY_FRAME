"""
1、计算光流并生成光流图，裁剪人脸区域；
2、保存人脸区域的RGB图和光流图，均为224*224

定义视频文件夹和结果保存文件夹路径
video_folder = "D:\\Deepfakes_datasets\\test\\opticalflow_video"
output_folder = "D:\\Deepfakes_datasets\\test\\opticalflow"
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_optical_flow(frame1, frame2):
    # 将帧转换为灰度图像
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def visualize_optical_flow(flow):
    # 计算光流位移向量的角度和大小
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 将角度映射到色调范围（0到180度）
    hue = angle * 180 / np.pi / 2

    # 将强度映射到饱和度范围（0到255）
    saturation = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # 创建HSV图像并将颜色编码的光流填充到其中
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = saturation
    hsv[..., 2] = 255

    # 将HSV图像转换为BGR图像
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def crop_face(frame, face_coordinates):
    if len(face_coordinates) > 0:
        x, y, w, h = face_coordinates
        face_image = frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (224, 224))
        return face_image
    return None

def make_opticalflow_figture():
    # 设置路径
    dataset_path = r'D:\Deepfakes_datasets\test\opticalflow_video'
    output_path = r'D:\Deepfakes_datasets\test\opticalflow'
    rgb_output_path = r'D:\Deepfakes_datasets\test\RGB'

    real_video_path = os.path.join(dataset_path, 'real')
    # fake_video_path = os.path.join(dataset_path, 'fake')

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(rgb_output_path, exist_ok=True)

    # 遍历真实视频目录
    for video_name in os.listdir(real_video_path):
        video_path = os.path.join(real_video_path, video_name)
        output_folder = os.path.join(output_path, 'real', video_name)
        rgb_output_folder = os.path.join(rgb_output_path, 'real', video_name)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(rgb_output_folder, exist_ok=True)

        # 获取视频总帧数
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 提取前10帧、中间10帧和最后10帧
        frame_indices = np.linspace(0, total_frames-1, num=30, dtype=int)
        frames = []
        for idx in frame_indices:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frames.append(frame)

        prev_frame = None
        for i, frame in enumerate(frames):
            if prev_frame is not None:
                # 计算光流
                flow = compute_optical_flow(prev_frame, frame)

                # 可视化光流
                flow_image = visualize_optical_flow(flow)

                # 获取当前帧及前后相邻两帧的人脸区域坐标
                prev_rgb_frame = frames[i-1]
                curr_rgb_frame = frame
                next_rgb_frame = frames[i+1] if i+1 < len(frames) else frame  # 修正超出索引范围的问题
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                boxes_prev = face_cascade.detectMultiScale(prev_rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                boxes_curr = face_cascade.detectMultiScale(curr_rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                boxes_next = face_cascade.detectMultiScale(next_rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # 确保每一帧都检测到人脸
                if len(boxes_prev) > 0 and len(boxes_curr) > 0 and len(boxes_next) > 0:
                    # 裁剪当前帧和前一帧的人脸区域
                    face_coordinates_prev = boxes_prev[0]
                    face_coordinates_curr = boxes_curr[0]
                    face_image_prev = crop_face(prev_rgb_frame, face_coordinates_prev)
                    face_image_curr = crop_face(curr_rgb_frame, face_coordinates_curr)

                    if face_image_prev is not None and face_image_curr is not None:

                        # 裁剪光流图像的人脸区域
                        flow_face_image = crop_face(flow_image, face_coordinates_curr)

                        if flow_face_image is not None:
                            # 调整裁剪后的光流图像的大小为224x224
                            resized_flow_face_image = cv2.resize(flow_face_image, (224, 224))

                            # 保存裁剪后的光流图像
                            output_face_filename = os.path.join(output_folder, f'flow_face_{i}.png')
                            cv2.imwrite(output_face_filename, resized_flow_face_image)

                            # 保存裁剪后的RGB帧
                            rgb_output_filename = os.path.join(rgb_output_folder, f'frame_{i}.png')
                            cv2.imwrite(rgb_output_filename, face_image_curr)

            prev_frame = frame
    print("保存完成.....")

make_opticalflow_figture()
