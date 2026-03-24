import cv2
import csv
import os
import numpy as np

csv_file = "qr_log.csv"

# CSVの準備（ヘッダー作成：4点分）
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["color_top", "color_right", "color_bottom", "color_left"])

cap = cv2.VideoCapture(0)
qrCodeDetector = cv2.QRCodeDetector()

last_saved_row = None

print("実行中... 各辺の内側4点の色の取得を開始します。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # QRコードの検出
    decodedText, points, _ = qrCodeDetector.detectAndDecode(frame)

    if points is not None:
        points = points[0]
        center_q = np.mean(points, axis=0) # QRの重心
        
        hex_colors = []
        point_coords = []

        # 4つの辺に対して処理 (0-1, 1-2, 2-3, 3-0)
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            
            # 1. 辺の中点を計算
            mid_point = (p1 + p2) / 2
            
            # 2. 中点から重心（中心）に向かうベクトルを計算
            # このベクトル方向に少し動かすことで「内側」に入る
            # 係数 0.15 はファインダーの半分程度を想定した調整値
            vec_to_center = center_q - mid_point
            target_point = mid_point + (vec_to_center * 0.15)
            
            tx, ty = int(target_point[0]), int(target_point[1])
            point_coords.append((tx, ty))

            # 3. 色取得
            b, g, r = frame[ty, tx]
            hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))

        # --- 描画処理 ---
        for i, (tx, ty) in enumerate(point_coords):
            cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1) # 赤い点で計測箇所を表示
            # 枠線の描画
            start_p = tuple(map(int, points[i]))
            end_p = tuple(map(int, points[(i + 1) % 4]))
            cv2.line(frame, start_p, end_p, (0, 255, 0), 2)

        # --- CSV保存処理 ---
        if hex_colors != last_saved_row:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(hex_colors)
                print(f"Saved 4 points: {hex_colors}")
            last_saved_row = hex_colors

    cv2.imshow('QR 4-Point Color Picker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
