import cv2
import csv
import os
import numpy as np

csv_file = "qr_log.csv"

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "C00","C01","C02",
            "C10","C11","C12",
            "C20","C21","C22"
        ])

# --- 設定 ---
SAMPLE_RADIUS = 3   # 色サンプリング領域の半径（px）
HOLD_FRAMES   = 10  # 検出できないとき前回座標を保持するフレーム数

cap = cv2.VideoCapture(0)
qrCodeDetector = cv2.QRCodeDetector()

last_pts     = None
hold_counter = 0

print("実行中...")


def quad_interp(p0, p1, p2, p3, u, v):
    P = (
        np.array(p0) * (1-u)*(1-v) +
        np.array(p1) * u*(1-v) +
        np.array(p2) * u*v +
        np.array(p3) * (1-u)*v
    )
    return float(P[0]), float(P[1])


def get_inner_corner(p0, p1, p2, p3, pos, version=1):
    n = 17 + 4 * version
    coords = {
        'TL': (5.0/n,       5.0/n),
        'TR': ((n-5.0)/n,   5.0/n),
        'BR': ((n-5.0)/n,   (n-5.0)/n),
        'BL': (5.0/n,       (n-5.0)/n),
    }
    u, v = coords[pos]
    return quad_interp(p0, p1, p2, p3, u, v)


def sample_color(frame, cx, cy, radius):
    """周辺領域の中央値で色取得（外れ値・ノイズに強い）"""
    h, w = frame.shape[:2]
    x1 = max(0, int(cx) - radius)
    x2 = min(w-1, int(cx) + radius)
    y1 = max(0, int(cy) - radius)
    y2 = min(h-1, int(cy) + radius)
    region = frame[y1:y2+1, x1:x2+1]
    if region.size == 0:
        px = frame[max(0,min(int(cy),h-1)), max(0,min(int(cx),w-1))]
        return int(px[2]), int(px[1]), int(px[0])
    return (
        int(np.median(region[:,:,2])),
        int(np.median(region[:,:,1])),
        int(np.median(region[:,:,0])),
    )


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    decodedText, points, _ = qrCodeDetector.detectAndDecode(gray)

    detected_now = points is not None

    if detected_now:
        last_pts     = points[0]
        hold_counter = HOLD_FRAMES
    else:
        hold_counter -= 1

    if last_pts is not None and hold_counter > 0:
        p0, p1, p2, p3 = last_pts

        cv2.polylines(frame, [last_pts.astype(int)], True, (0,255,0), 1)

        c_TL = get_inner_corner(p0, p1, p2, p3, 'TL')
        c_TR = get_inner_corner(p0, p1, p2, p3, 'TR')
        c_BR = get_inner_corner(p0, p1, p2, p3, 'BR')
        c_BL = get_inner_corner(p0, p1, p2, p3, 'BL')

        quad = np.array([c_TL, c_TR, c_BR, c_BL], dtype=np.float32).astype(int)
        cv2.polylines(frame, [quad], True, (0,165,255), 2)

        hex_colors = []
        for i in range(3):
            for j in range(3):
                u = (j + 0.5) / 3
                v = (i + 0.5) / 3
                cx, cy = quad_interp(c_TL, c_TR, c_BR, c_BL, u, v)
                r, g, b = sample_color(frame, cx, cy, SAMPLE_RADIUS)
                hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))
                cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

        if detected_now:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(hex_colors)

        if decodedText:
            cv2.putText(frame, decodedText[:40], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('QR Center 9-Grid', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
