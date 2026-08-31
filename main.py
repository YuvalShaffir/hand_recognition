import cv2 

capture = cv2.VideoCapture(index=0)

while True:
    ok, frame = capture.read()
    if not ok:
        break
    cv2.imshow("camera", frame)

    # Start OpenCV event loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()