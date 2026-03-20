import cv2
import numpy as np

qrCodeDetector = cv2.QRCodeDetector()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    ret,frame = cap.read()
    if not ret:
        break

    decodedText, points,_=qrCodeDetector.detectAndDecode(frame)

    if points is not None:
        pts = points[0].astype(int)

        cv2.polylines(frame,[pts],isClosed=True, color=(0,255,0),thickness=3)

        if decodedText:
            cv2.putText(frame,decodedText,tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
            print(f"Decoded: {decodedText}")


    cv2.imshow('QR Code Scanner',frame )

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
