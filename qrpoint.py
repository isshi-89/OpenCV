import cv2
import csv
import os
import numpy as np

csv_file = "qr_log.csv"

# CSVの準備
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["C00","C01","C02","C10","C11","C12","C20","C21","C22"])

cap = cv2.VideoCapture(0)
qrCodeDetector = cv2.QRCodeDetector()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

last_saved_row = None

print("実行中... 透視変換による高精度モードで開始します。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 検出用にグレー化とコントラスト調整
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)

    # QRコード検出
    decodedText, points, _ = qrCodeDetector.detectAndDecode(enhanced)

    if points is not None:
        # 検出された4角の座標 (p0:左上, p1:右上, p2:右下, p3:左下)
        src_pts = points[0].astype(np.float32)

        # 変換後の正方形サイズ（300x300pxの仮想平面を作る）
        side = 300
        dst_pts = np.array([
            [0, 0],
            [side - 1, 0],
            [side - 1, side - 1],
            [0, side - 1]
        ], dtype=np.float32)

        # 透視変換行列を取得
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 逆行列（仮想平面上の座標を元の画像上の座標に戻すために使用）
        M_inv = np.linalg.inv(M)

        hex_colors = []
        
        # 3x3の各セルの中心を計算
        for i in range(3):   # 縦(v)
            for j in range(3): # 横(u)
                # 仮想平面(300x300)上での中心点
                v_x = (j + 0.5) * (side / 3)
                v_y = (i + 0.5) * (side / 3)
                
                # 仮想平面の点を元の画像上の座標に変換
                target_pt = np.array([[[v_x, v_y]]], dtype=np.float32)
                map_pt = cv2.perspectiveTransform(target_pt, M_inv)[0][0]
                
                cx, cy = int(map_pt[0]), int(map_pt[1])

                # 画像範囲内かチェック
                h, w = frame.shape[:2]
                cx = max(2, min(cx, w - 3))
                cy = max(2, min(cy, h - 3))

                # --- 精度向上のためのポイント：周辺5x5マスの平均色を取得 ---
                roi = frame[cy-2:cy+3, cx-2:cx+3]
                if roi.size > 0:
                    avg_bgr = np.mean(roi, axis=(0, 1))
                    b, g, r = avg_bgr.astype(int)
                else:
                    b, g, r = frame[cy, cx]

                hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))
                
                # 描画（確認用）
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # 変化があった場合のみCSV保存
        if hex_colors != last_saved_row:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(hex_colors)
            last_saved_row = hex_colors
            print("Saved row:", hex_colors[0], "...")

    cv2.imshow('High-Precision QR Color Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
