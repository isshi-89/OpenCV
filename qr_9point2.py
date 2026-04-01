import cv2
import csv
import os
import numpy as np

# -------------------------------------------------------
# 設定
# -------------------------------------------------------
CSV_FILE = "qr_log.csv"
SIDE = 300          # 透視変換後の仮想平面サイズ (px)
PATCH = 2           # 色取得に使う周辺パッチ半径 (5x5)

# -------------------------------------------------------
# 透視変換後の目標座標（固定なのでループ外で定義）
# -------------------------------------------------------
DST_PTS = np.array([
    [0,          0         ],
    [SIDE - 1,   0         ],
    [SIDE - 1,   SIDE - 1  ],
    [0,          SIDE - 1  ],
], dtype=np.float32)

# 3×3セル中心の仮想座標も固定なのでループ外で計算
CELL_CENTERS_VIRTUAL = np.array(
    [[[( j + 0.5) * (SIDE / 3), (i + 0.5) * (SIDE / 3)]]
     for i in range(3) for j in range(3)],
    dtype=np.float32
)   # shape: (9, 1, 2)

# -------------------------------------------------------
# CSVヘッダー作成（未存在時のみ）
# -------------------------------------------------------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(
            ["C00","C01","C02","C10","C11","C12","C20","C21","C22"]
        )

# -------------------------------------------------------
# カメラ・検出器初期化
# -------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("カメラを開けませんでした")

qr_detector = cv2.QRCodeDetector()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("実行中... 'q' キーで終了")

# -------------------------------------------------------
# ファイルはループ外で一度だけ開く
# -------------------------------------------------------
try:
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as csv_f:
        writer = csv.writer(csv_f)
        last_saved_row: tuple | None = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("フレーム取得失敗 — 終了します")
                break

            h, w = frame.shape[:2]

            # --- 検出用前処理（グレー化 + CLAHE）---
            # ※色取得は元のカラー画像 frame から行う（色情報を保持するため）
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)

            # --- QRコード検出 ---
            decoded_text, points, _ = qr_detector.detectAndDecode(enhanced)

            if points is not None:
                src_pts = points[0].astype(np.float32)

                # 透視変換行列（src→dst）と逆行列（dst→src）
                # 逆行列は引数を入れ替えるだけで直接取得できる
                M_inv = cv2.getPerspectiveTransform(DST_PTS, src_pts)

                # 仮想平面上のセル中心 → 元画像座標に一括変換
                real_pts = cv2.perspectiveTransform(
                    CELL_CENTERS_VIRTUAL, M_inv
                )  # shape: (9, 1, 2)

                hex_colors = []

                for pt in real_pts:
                    cx = int(np.clip(pt[0][0], PATCH, w - PATCH - 1))
                    cy = int(np.clip(pt[0][1], PATCH, h - PATCH - 1))

                    # 5×5パッチの平均色（照明ノイズを軽減）
                    roi = frame[cy - PATCH: cy + PATCH + 1,
                                cx - PATCH: cx + PATCH + 1]
                    avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
                    b, g, r = avg_bgr
                    hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))

                    # 確認用マーカー描画
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # QRコード内容をフレームに表示
                if decoded_text:
                    cv2.putText(frame, decoded_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2)

                # 変化があった場合のみ保存
                row_tuple = tuple(hex_colors)
                if row_tuple != last_saved_row:
                    writer.writerow(hex_colors)
                    csv_f.flush()           # OSバッファを即時書き込み
                    last_saved_row = row_tuple
                    print("Saved:", hex_colors)

            cv2.imshow('QR Color Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Ctrl+C で中断されました")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("終了しました")
