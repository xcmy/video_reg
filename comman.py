import cv2
import imutils
 
# 加载摄像头
cap = cv2.VideoCapture(0)
print(cv2.getBuildInformation())

if not cap.isOpened():
    print("无法打开摄像头")
 
# 创建人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
# 初始化人数计数器
num_people = 0
 
while True:
    # 读取摄像头数据
    ret, frame = cap.read()
    
    ret, frame = cap.read()
    if frame is None:
        print("没有捕获到图像")
    else:
        # 处理图像
        print(frame.shape)
 
    # 调整图像大小
    frame = imutils.resize(frame, width=500)
 
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    # 更新人数计数器
    num_people = len(faces)
 
    # 在图像上显示人数
    cv2.putText(frame, "Number of People: {}".format(num_people), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    # 显示图像
    cv2.imshow('frame', frame)
 
    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
# 释放摄像头并关闭窗口

cap.release()
cv2.destroyAllWindows()