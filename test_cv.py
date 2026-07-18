import cv2

cap = cv2.VideoCapture(0)  # 0 是默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    cv2.imshow("Camera", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

