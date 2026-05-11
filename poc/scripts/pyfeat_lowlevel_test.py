#!/usr/bin/env python3
"""py-feat 低层 API 测试 — 跳过文件 I/O，直接传 numpy array。

如果 detect_image (高层 API) NO FACE，试试低层 detect_faces + detect_landmarks + detect_aus。
30 秒摄像头采样，对比两种 API 检测成功率。
"""
from __future__ import annotations

import time
import warnings

import cv2

warnings.filterwarnings("ignore")

import numpy as np
import torch


def main() -> None:
    print("\n  ⏳ 加载 py-feat detector...")
    t0 = time.time()
    from feat import Detector  # type: ignore
    detector = Detector(au_model="xgb")
    print(f"  ✅ {time.time()-t0:.1f}s")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头打不开"); return

    print("  📷 摄像头打开。20 秒测试 — 自然脸 + 微笑 + 皱眉")

    n_total = 0
    n_high_ok = 0
    n_low_ok = 0
    last = 0.0
    t_start = time.time()

    try:
        while time.time() - t_start < 20:
            now = time.time()
            if now - last < 1.0:
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                continue
            last = now
            ret, frame = cap.read()
            if not ret:
                continue
            n_total += 1

            # 测高层 API
            tmp_path = "/tmp/pyfeat_hltest.jpg"
            cv2.imwrite(tmp_path, frame)
            try:
                fex_high = detector.detect_image([tmp_path], output_size=512)
                if fex_high is not None and len(fex_high) > 0:
                    n_high_ok += 1
            except Exception as e:
                pass

            # 测低层 API
            try:
                # py-feat 0.6.2 detect_faces 接受 numpy array (B, H, W, 3) 或 (H, W, 3)
                # 注：需要 RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 加 batch 维度
                rgb_batch = np.expand_dims(rgb, axis=0)
                rgb_tensor = torch.from_numpy(rgb_batch).permute(0, 3, 1, 2).float() / 255.0
                faces = detector.detect_faces(rgb_tensor)
                if faces is not None and len(faces) > 0 and len(faces[0]) > 0:
                    n_low_ok += 1
            except Exception as e:
                if n_total <= 2:
                    print(f"    low-level error: {e}")

            print(f"  #{n_total}: high_api={'OK' if n_high_ok > 0 and n_high_ok == n_total else 'FAIL'} low_api={'OK' if n_low_ok > 0 and n_low_ok == n_total else 'FAIL'}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n  📊 总结：{n_total} 帧采样")
    print(f"     high_api (detect_image)  成功率: {n_high_ok}/{n_total} = {n_high_ok/max(n_total,1):.0%}")
    print(f"     low_api  (detect_faces)  成功率: {n_low_ok}/{n_total} = {n_low_ok/max(n_total,1):.0%}")


if __name__ == "__main__":
    main()
