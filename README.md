# Traffic-Volume--ODYOLO11-Tracking
I had a wonderful summer break to do the things I love
## 📺 Demo Video

[![Watch the demo](https://img.youtube.com/vi/NI1ayaz5G3k/0.jpg)](https://www.youtube.com/watch?v=NI1ayaz5G3k)
# 🚗 Real-Time Object Detection and Tracking with YOLO11n

This project demonstrates how to combine **YOLO11n** for object detection with a **tracking algorithm** (e.g., SORT) to track and count objects (like vehicles) in video — **without using any external object counting library**.

---

## 📌 Features

- ✅ Real-time object detection using **YOLO11n**
- ✅ Object tracking using **SORT (Simple Online and Realtime Tracking)**
- ✅ Custom line-based object counting logic
- ✅ Supports video input
- ✅ Visualizations with bounding boxes, track IDs, and counting line
- 🚫 No use of `ObjectCounting` libraries — all logic is implemented manually for flexibility


## 🧠 How It Works

1. **YOLO11n** detects objects (e.g., cars) in each frame.
2. **SORT** assigns unique IDs to each object across frames.
3. A custom **counting line** is drawn in the frame.
4. When an object crosses the line in a specific direction, the count is incremented.
5. The results (bounding boxes, tracking IDs, count) are displayed and optionally saved.

---
