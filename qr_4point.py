import cv2
import csv
import os
import numpy as np

csv_file = "qr_log.csv"

# CSVの準備
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["color_top", "color_right", "color_bottom", "color_left"])

cap = cv2.VideoCapture(0)
qrCodeDetector = cv2.QRCodeDetector()

# コントラスト強調器の生成（一度だけ作成して使い回す）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

last_saved_row = None

print("実行中... グレー化前処理を適用して開始します。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- 精度向上のための前処理 ---
    # 1. グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. コントラスト強調（暗い場所や反射に強くなる）
    enhanced = clahe.apply(gray)

    # 検知には「加工後の画像（enhanced）」を使用
    decodedText, points, _ = qrCodeDetector.detectAndDecode(enhanced)

    if points is not None:
        points = points[0]
        # 重心計算
        center_q = np.mean(points, axis=0)
        
        hex_colors = []
        point_coords = []

        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            
            # 辺の中点
            mid_point = (p1 + p2) / 2
            
            # 重心方向に 15% 寄せる（ファインダーの内側を狙う）
            vec_to_center = center_q - mid_point
            target_point = mid_point + (vec_to_center * 0.15)
            
            tx, ty = int(target_point[0]), int(target_point[1])
            
            # 画像の範囲外チェック（エラー防止）
            h, w = frame.shape[:2]
            tx = max(0, min(tx, w - 1))
            ty = max(0, min(ty, h - 1))
            
            point_coords.append((tx, ty))

            # 色取得は「元のカラー画像（frame）」から！
            b, g, r = frame[ty, tx]
            hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))

        # --- 描画処理 ---
        for i, (tx, ty) in enumerate(point_coords):
            # 計測点を描画
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1) 
            # 外枠を描画
            start_p = tuple(map(int, points[i]))
            end_p = tuple(map(int, points[(i + 1) % 4]))
            cv2.line(frame, start_p, end_p, (0, 255, 0), 2)

        # --- CSV保存処理 ---
        if hex_colors != last_saved_row:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(hex_colors)
                print(f"Saved: {hex_colors}")
            last_saved_row = hex_colors

    # プレビューはカラー画像を表示
    cv2.imshow('QR 4-Point High-Precision', frame)
    
    # オプション：検知用の白黒画像を見たい場合はコメント解除
    # cv2.imshow('Detection View', enhanced)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
