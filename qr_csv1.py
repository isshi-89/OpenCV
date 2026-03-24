import cv2
import csv
import os
import numpy as np

csv_file = "qr_log.csv"

# CSVの準備（ヘッダー作成）
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["hex_color"])

cap = cv2.VideoCapture(0)
qrCodeDetector = cv2.QRCodeDetector()

# 前回の色を記憶する変数（重複保存を防ぐため）
last_saved_color = None

print("実行中... 'q'キーで終了します。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # QRコードの検出とデコード
    decodedText, points, _ = qrCodeDetector.detectAndDecode(frame)

    if points is not None:
        points = points[0]
        
        # QRコードの中心座標を計算
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))

        # 指定座標のB,G,Rを取得
        b, g, r = frame[center_y, center_x]
        # 16進数カラーコードに変換
        hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

        # --- 視覚的なフィードバック ---
        # QRコードを囲む枠線を描画
        for i in range(len(points)):
            start_point = tuple(map(int, points[i]))
            end_point = tuple(map(int, points[(i + 1) % len(points)]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

        # 中心にドットを描画
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        # カラーコードを表示
        cv2.putText(frame, hex_color, (center_x + 10, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- CSV保存処理 ---
        # 「前回の色と違う」かつ「色が取得できている」場合のみ保存
        if hex_color != last_saved_color:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([hex_color])
                print(f"Saved: {hex_color}")
            
            # 今保存した色を記憶
            last_saved_color = hex_color

        # コンソールにも現在の色を表示
        # print(f"Color: {hex_color} at ({center_x},{center_y})")

    # ウィンドウ表示
    cv2.imshow('QR Color Picker', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
