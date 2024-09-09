from ultralytics import YOLO
import cv2
import imutils

# count people id

# 加载摄像头
cap = cv2.VideoCapture(0)


# Load a pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # 调整图像大小
    frame = imutils.resize(frame, width=500)

    # Pass the frame to YOLO model for inference
    results = model(frame)
    

    # Visualize results on the frame
    result_frame = results[0].plot()  # Draw bounding boxes and labels on the frame

    # Display the resulting frame
    cv2.imshow('YOLOv10 Real-Time Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

 
# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()



# # Perform object detection on an image
# results = model("./3.jpg")

# # Display the results
# results[0].show()