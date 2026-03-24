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
        points = points[0]

        print(f"Points: {points}")

        if len(points) == 4 and all(len(point) == 2 for point in points):
            for i in range(len(points)):
                start_point = tuple(map(int,points[i]))
                end_point = tuple(map(int,points[(i + 1) % len(points)]))

                cv2.line(frame, start_point,end_point, (0,255,0),3)
                
        else:
             print("Invalid points structure.")

        print(decodedText)
                
        


    cv2.imshow('QR Code Scanner',frame )

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
