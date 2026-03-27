import cv2
import csv
import os
import numpy as np

csv_file = "qr_log.csv"

# CSVの準備（9セルを横並びで保存）
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "C00","C01","C02",
            "C10","C11","C12",
            "C20","C21","C22"
        ])

cap = cv2.VideoCapture(0)
qrCodeDetector = cv2.QRCodeDetector()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

last_saved_row = None

print("実行中... グレー化前処理を適用して開始します。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)

    decodedText, points, _ = qrCodeDetector.detectAndDecode(enhanced)

    if points is not None:
        points = points[0]  # p0,p1,p2,p3

        p0, p1, p2, p3 = points

        # 四角形内部補間関数
        def quad_interp(u, v):
            P = (
                p0 * (1-u)*(1-v) +
                p1 * u*(1-v) +
                p2 * u*v +
                p3 * (1-u)*v
            )
            return int(P[0]), int(P[1])

        hex_colors = []
        cell_points = []

        # 3×3 の 9セルの中心を取得（横並び順）
        for i in range(3):        # v方向（縦）
            for j in range(3):    # u方向（横）
                u = (j + 0.5) / 3
                v = (i + 0.5) / 3

                cx, cy = quad_interp(u, v)

                # 範囲チェック
                h, w = frame.shape[:2]
                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))

                cell_points.append((cx, cy))

                # 色取得
                b, g, r = frame[cy, cx]
                hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))

        # 描画（9点）
        for (cx, cy) in cell_points:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # CSV保存（横並び9セル）
        if hex_colors != last_saved_row:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(hex_colors)
                print(f"Saved: {hex_colors}")
