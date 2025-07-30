from ultralytics import YOLO
import cv2
import cvzone
import numpy as np 
import math
import sort
model = YOLO("./model/yolo11m.pt")

cap = cv2.VideoCapture('./video/jaffiic_japan.mp4')  # video


tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('./outputVideo/output_yolo_2.mp4', fourcc, fps, (1280,720))

class_name = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "brocolli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush"
  }
limits = [700, 664, 863, 658]
limits_left = [455, 658, 649, 658]
total_Count_right = []
total_Count_left = []

mask = cv2.imread('./img/JPcanva_4.png')
while True:
    ret, frame = cap.read()
    #resize frame
    frame = cv2.resize(frame,(1280,720))
    #logo
    img_graphic = cv2.imread('./img/logo.png',cv2.IMREAD_UNCHANGED)
    img_graphic = cv2.cvtColor(img_graphic, cv2.COLOR_BGR2BGRA)
    img_graphic =cv2.resize(img_graphic,(100,100))
    cv2.putText(frame, text='traffic volume G025C1120', org=(0, 130), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0, 255, 0), thickness=2)
    frame = cvzone.overlayPNG(frame,img_graphic,(0,0))
    if not ret:
        break
    #mask
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    img_region = cv2.bitwise_and(frame, mask)
    detections = np.empty((0, 5))


#fake zone 
    overlay = frame.copy()

    # Danh sách tọa độ và màu tương ứng
    regions = [
        ([(553, 558), (786, 564), (880, 655), (449, 655)], (0, 255, 0)),  # xanh
        ([(880, 656), (945, 712), (392, 716), (447, 656)], (0, 255, 255)),    # vang
        ([(549, 558), (639, 510), (726, 512), (786, 562)], (0, 0, 255))     # Đỏ
    ]

    # Vẽ từng vùng overlay
    for points, color in regions:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color=color)

    # Trộn overlay với frame gốc theo alpha (độ trong suốt)
    alpha = 0.3  # Có thể chỉnh độ mờ
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Tùy chọn: bo viền vùng cho rõ
    for points, color in regions:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    results = model(img_region, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0]
            conf_int = math.ceil((box.conf[0] * 100)) / 100
            cls = box.cls[0]
            class_index = str(int(cls))
            cls_name = class_name[class_index]
            if (cls_name == 'motorcycle'
                or cls_name == 'car'
                or cls_name == 'truck'
                or cls_name == 'bus') \
                and conf_int > 0.1:
                # cvzone.putTextRect(frame, f'{cls_name} {conf_int} ', (max(0, x1), max(40, y1)), scale=0.5, thickness=1)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=2, t=1, rt=5, colorR=(255, 0, 255), colorC=(0, 255, 0))
                cx, cy = x1 + w // 2, y1 + h // 2
                # cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                current_array = np.array([x1, y1, x2, y2, conf_int])
                detections = np.vstack((detections, current_array))
    result_tracker = tracker.update(detections)
    for result in result_tracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1


        

#main line

        # cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        # cv2.line(frame, (limits_left[0], limits_left[1]), (limits_left[2], limits_left[3]), (0, 0, 255), 5)

        # cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255), colorC=(0, 255, 0),t=1)
        # cvzone.putTextRect(frame, f'{int(Id)} ', (max(0, x1), max(35, y1)), scale=3, thickness=1, offset=3)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        color_detected = (0, 0, 255)

        if (limits[0] < cx < limits[2]) and (limits[1] - 5 < cy < limits[3] + 5):
            color_detected = (0, 255, 0)
            if total_Count_right.count(Id) == 0 : 
                total_Count_right.append(Id)
                

                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
                cv2.putText(frame, 'Detected', (limits[0], limits[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(0, 255, 0), thickness=2)
            

        
        if (limits_left[0] < cx < limits_left[2]) and (limits_left[1] - 5 < cy < limits_left[3] + 5):
            color_detected = (0, 255, 0)
            if total_Count_left.count(Id) == 0 : 
                total_Count_left.append(Id)
                
                cv2.line(frame, (limits_left[0], limits_left[1]), (limits_left[2], limits_left[3]), (0, 0, 255), 5)
                cv2.putText(frame, 'Detected', (limits_left[0],limits_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(0, 255, 0), thickness=2)
              
        cv2.putText(frame, 'Detected', (23, 178), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=color_detected, thickness=2)
        cv2.putText(frame,f'{len(total_Count_right)} ', (779, 656), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), thickness=2)        
        cv2.putText(frame,f'{len(total_Count_left)} ', (558, 656), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), thickness=2)        
        # cvzone.putTextRect(frame, f'{total_Count_right} ', (779, 656), scale=3, thickness=2, offset=10)

    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('region', img_region)
    # cv2.imwrite('./img/japan_1.jpg', frame) #save img test
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()