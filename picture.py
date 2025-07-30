from ultralytics import YOLO
import cv2
import cvzone
import numpy as np 
import math
import sort
model = YOLO("./model/yolo11m.pt")

cap = cv2.VideoCapture('./video/jaffiic_japan.mp4')  # video

while True:
    ret,frame = cap.read()

    # cv2.imshow('web cam',frame) 
    # out.write(frame)  # Ghi frame đã chỉnh sửa vào file outpu
    cv2.imwrite('./img/japan_2.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()