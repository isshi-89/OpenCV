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
print("実行中... ファインダー内側正方形を基準にQR中央エリアを9分割します。")

def get_finder_inner_center(p0, p1, p2, p3, finder_pos, version=1):
    """
    ファインダーパターン内側3×3正方形の「中心点」を返す。
    内側正方形 = ファインダー内のモジュール2〜4（0-indexed）の中心
    → モジュール3.0 / n_modules が中心の正規化座標
    """
    n = 17 + 4 * version  # バージョン1=21モジュール

    if finder_pos == 'TL':
        # 左上ファインダーの内側正方形中心 = モジュール(3, 3)
        u = 3.5 / n
        v = 3.5 / n
    elif finder_pos == 'TR':
        # 右上ファインダーの内側正方形中心 = モジュール(n-4, 3)
        u = (n - 3.5) / n
        v = 3.5 / n
    elif finder_pos == 'BL':
        # 左下ファインダーの内側正方形中心 = モジュール(3, n-4)
        u = 3.5 / n
        v = (n - 3.5) / n
    else:
        return None

    # 双線形補間でピクセル座標を計算
    P = (
        np.array(p0) * (1-u)*(1-v) +
        np.array(p1) * u*(1-v) +
        np.array(p2) * u*v +
        np.array(p3) * (1-u)*v
    )
    return (int(P[0]), int(P[1]))

def get_finder_inner_corner(p0, p1, p2, p3, finder_pos, version=1):
    """
    ファインダー内側正方形の「QR中央側の角」を返す。
    これを4頂点として中央エリアを定義する。

    内側正方形はモジュール2〜4の範囲。
    QR中央側の角 = 内側正方形のQR内側の角
      TL → 右下角 = モジュール(5, 5)
      TR → 左下角 = モジュール(n-5, 5)
      BL → 右上角 = モジュール(5, n-5)
      BR（推定）= モジュール(n-5, n-5)  ← QRコードには存在しないので推定
    """
    n = 17 + 4 * version

    if finder_pos == 'TL':
        u, v = 5.0 / n, 5.0 / n
    elif finder_pos == 'TR':
        u, v = (n - 5.0) / n, 5.0 / n
    elif finder_pos == 'BL':
        u, v = 5.0 / n, (n - 5.0) / n
    elif finder_pos == 'BR':
        # 右下にはファインダーがないので対称的に推定
        u, v = (n - 5.0) / n, (n - 5.0) / n
    else:
        return None

    P = (
        np.array(p0) * (1-u)*(1-v) +
        np.array(p1) * u*(1-v) +
        np.array(p2) * u*v +
        np.array(p3) * (1-u)*v
    )
    return (int(P[0]), int(P[1]))

def quad_interp_pts(corner_TL, corner_TR, corner_BR, corner_BL, u, v):
    """4頂点の四角形内を双線形補間"""
    P = (
        np.array(corner_TL) * (1-u)*(1-v) +
        np.array(corner_TR) * u*(1-v) +
        np.array(corner_BR) * u*v +
        np.array(corner_BL) * (1-u)*v
    )
    return int(P[0]), int(P[1])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    decodedText, points, _ = qrCodeDetector.detectAndDecode(gray)

    if points is not None:
        pts = points[0]  # p0(左上), p1(右上), p2(右下), p3(左下)
        p0, p1, p2, p3 = pts

        # QRコード全体の輪郭
        cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 1)

        version = 1  # デフォルトv1（必要に応じて推定ロジックを追加）

        # ファインダー内側正方形の「QR中央側の角」を4点取得
        c_TL = get_finder_inner_corner(p0, p1, p2, p3, 'TL', version)
        c_TR = get_finder_inner_corner(p0, p1, p2, p3, 'TR', version)
        c_BR = get_finder_inner_corner(p0, p1, p2, p3, 'BR', version)  # 推定
        c_BL = get_finder_inner_corner(p0, p1, p2, p3, 'BL', version)

        # 中央エリアの輪郭を描画
        center_quad = np.array([c_TL, c_TR, c_BR, c_BL], dtype=np.int32)
        cv2.polylines(frame, [center_quad], True, (0, 165, 255), 2)

        # ファインダー内側正方形の角を描画
        for pt, label in [(c_TL,'TL'),(c_TR,'TR'),(c_BR,'BR'),(c_BL,'BL')]:
            cv2.circle(frame, pt, 6, (255, 0, 255), -1)

        # 中央エリアを3×3に9分割して色取得
        hex_colors = []
        cell_points = []

        for i in range(3):      # v方向（縦）
            for j in range(3):  # u方向（横）
                u = (j + 0.5) / 3
                v = (i + 0.5) / 3
                cx, cy = quad_interp_pts(c_TL, c_TR, c_BR, c_BL, u, v)

                # 範囲クリップ
                h, w = frame.shape[:2]
                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))
                cell_points.append((cx, cy))

                b, g, r = frame[cy, cx]
                hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))

        # 9点を描画
        for idx, (cx, cy) in enumerate(cell_points):
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(idx), (cx+6, cy-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # CSV保存
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(hex_colors)

        if decodedText:
            cv2.putText(frame, decodedText[:40], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('QR Center Area 9-Grid', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
