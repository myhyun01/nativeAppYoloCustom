import sys
import cv2
import torch
import time
# 여기서 사용한 Pyside6는 pyQt5와 완전 호환됩니다. 
# GPT에 pyQt5와 PySide6는 뭐가 다르냐?  물어 보세요 
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer
import serial
from gtts import gTTS
import pygame
import os

# import pathlib  # PosixPath 오류 해결
# from pathlib import Path  # PosixPath 오류 해결
# pathlib.PosixPath = pathlib.WindowsPath  # PosixPath 오류 해결

# 작업폴더에 Yolov5를 클로닝하세요 
# git clone https://github.com/ultralytics/yolov5.git

# window_yolo_ui.ui 파일을 window_yolo_ui.py 파일로 변환 하여 import 시키는 방법
from window_yolo_ui import Ui_MainWindow
# 터미널에서 pyside6-uic window_yolo.ui -o window_yolo_ui.py

# class name을 갖고 있은 배열을 이용하기 
# COCO 클래스 이름
cls_names = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# # 창배군이 만든 모델의 클래스 이름
# cls_names = [
#     'fall_down',"normal"
#  ]


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 메인 윈도우 제목 설정
        self.setWindowTitle("Custom Model을 이용한 YOLO 웹캠")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 자신이 만든(학습시킨) pt파일을(대부분은 best.py라고 쓰고 있겠지만) 이름을 바꿔서 
        # 내 폴더에 놓고 사용하는 방법
        self.model = torch.hub.load("./yolov5", model="custom",
                                    path="./myModels/yolov5s.pt", source="local")
        # 폴더 
        # 여기는 반드시 model = "custom"
        # 그리고 path 에 있는 것이 실제 본인의 모델임 여기서는 편의상 yolov5s.pt를 사용했음.(본인것으로 바꾸세요)
        # 맨뒤에 source = "local"

        self.model.to(device=self.device)

        # 타이머를 이용해서 20mSec 마다 1개의 프레임을 읽어 오는 것 
        # 이것은 Thread를 쓰는 것 보다 비교적 간단한 방법으로 화면 딜레이를 줄일수 있음.,
        # 즉 모델이 Predict 하는 시간을 벌어 주는 역활을 함.         
        self.timer = QTimer()  # 타이머
        self.timer.setInterval(100)  # 타이머 간격을 20ms로 설정, 즉 매 20ms마다 신호를 보내 한 프레임 추론
                                    # 실제 처리 시간이 167 정도 소요 되므로 인터벌을 길게 잡아도 될듯 
        
        self.video = cv2.VideoCapture(0)  # 웹캠 열기

        self.timer.timeout.connect(self.video_pred) # 매 타이머 마다 이 함수를 실행하도록 예약 
        self.timer.start()

        # 직렬 포트 초기화
        self.ser = serial.Serial('COM8', 115200)  # 포트와 속도는 환경에 맞게 설정 (Windows에서는 COM 포트 사용, Linux/Mac에서는 '/dev/ttyUSB0' 등 사용)

        # pygame 초기화
        pygame.mixer.init()

        # 프로그램 카메라 시작 대기 
        time.sleep(2)
        self.speak(f"로봇이 준비 되었습니다.")



    def convert2QImage(self, img):  # 배열을 QImage로 변환하여 표시
        height, width, channel = img.shape
        return QImage(img, width, height, width * channel, QImage.Format_RGB888)

    def video_pred(self):  # 비디오 감지
        ret, frame = self.video.read()  # ret은 프레임을 읽었는지 여부, frame은 읽은 프레임
        # 매번 read() 시 다음 프레임을 읽음
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.input.setPixmap(QPixmap.fromImage(self.convert2QImage(frame)))
            start = time.perf_counter()
            results = self.model(frame)
            
            # 인식 결과를 먼저 Display 함 
            image = results.render()[0]
            self.output.setPixmap(
                QPixmap.fromImage(self.convert2QImage(image)))
            
            end = time.perf_counter()
            self.label_3.setText(f'판독시간:{round((end - start) * 1000,4)} ms')

            # 감지된 객체를 confidence 값으로 정렬하여 상위 1개 출력
            detections = results.xyxy[0].cpu().numpy()
            if len(detections) > 0:
                sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)[:1]  # confidence 값 기준으로 정렬 후 상위 1개 선택
                for det in enumerate(sorted_detections):
                    xyxy = det[:4]
                    conf = det[4]
                    cls = int(det[5])
                    class_name = cls_names[cls] if cls < len(cls_names) else f"클래스 {cls}"
                    print(f'xyxy={xyxy}')
                                        
                    # 가장 Confidence 값이 높은 객체 1개만 인식하도록 했으므로 
                    self.label.setText(f"클래스:{cls}-{class_name}, confidence: {conf:.2f}")                                            
                    self.label_2.setText("")                    
                    # self.speak(f"Detected {class_name} with confidence {conf:.2f}")

                    # 인식된 결과에 따라서 로봇의 행동이나 음성을 지정
                    if class_name == 'person':
                        self.robotAction(17) #Motion Table에 정의된 17번동작 위너
                        self.speak(f"안녕하신가 휴먼. 무엇을 도와줄까? ")
                        time.sleep(7)
                        self.robotAction(1) #Motion Table에 정의된 1번동작 차렷
                        time.sleep(1)
                    if class_name == 'bottle':
                        self.robotAction(30) #Motion Table에 정의된 30번동작 zap 날리면 
                        self.speak(f"그 병안에는 어떤 맛있는 음료가 들어 있냐 하냐? ")
                        time.sleep(7)
                        self.robotAction(1) #Motion Table에 정의된 1번동작 차렷
                        time.sleep(1)

            
            
            
    
    def robotAction(self, no):
        # 로봇에게 명령을 전달하는 함수
        print(f"Robot Action: {no}")
        if self.ser.is_open:
            exeCmd = bytearray([0xff, 0xff, 0x4c, 0x53, 0x00,
                                0x00, 0x00, 0x00, 0x30, 0x0c, 0x03,
                                no, 0x00, 100, 0x00])

            # checksum 계산
            exeCmd[14] = sum(exeCmd[6:14]) & 0xFF

            # 명령어 전송
            self.ser.write(exeCmd)
            time.sleep(0.05) # 통신에 필요한 최소 시간 

    def closeEvent(self, event):
        # 애플리케이션 종료 시 직렬 포트를 닫는다
        if self.ser.is_open:
            self.ser.close()
        event.accept()

    def speak(self, text):
        self.timer.stop()
        tts = gTTS(text=text, lang='ko')
        tts.save("output.mp3")
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove("output.mp3")        
        self.timer.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())