import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("./video/jaffiic_japan.mp4")
assert cap.isOpened(), "Error reading video file"

# Định nghĩa 2 vùng đếm (rectangle region)
region_points_1 = [(1546, 1880), (1964, 1880), (1952, 2011), (1419, 1977)]
region_points_2 = [(2070, 1821), (2396, 1804), (2522, 1922), (2078, 1922)]

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("./outputVideo/japan_yolo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Khởi tạo 2 bộ đếm cho 2 vùng
counter1 = solutions.ObjectCounter(
    show=True,
    region=region_points_1,
    model="yolo11n.pt",
    classes=[1,2,3,5,6,7],
)
counter2 = solutions.ObjectCounter(
    show=True,
    region=region_points_2,
    model="yolo11n.pt",
    classes=[1,2,3,5,6,7],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results1 = counter1(im0)
    results2 = counter2(im0)

    # Vẽ kết quả lên frame
    im0 = results1.plot_im
    im0 = results2.plot_im


    video_writer.write(im0)
    cv2.imshow("YOLO Count", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()