#!/usr/bin/env python3
"""快速测试 py-feat 在真人摄像头帧上的 AU 输出。

跑 30 秒摄像头流，每秒提取一帧给 py-feat，输出 17 个 AU 实测值范围。

期望对比：
- 我们手写公式 brow_furrow max=0.00（bug）
- py-feat AU4 应该 0.5+（皱眉时）
"""
from __future__ import annotations

import time
import warnings

import cv2

warnings.filterwarnings("ignore")

import numpy as np
from feat import Detector


def main() -> None:
    print("\n  ⏳ 加载 py-feat detector（首次会下载模型 ~500MB，请耐心）...")
    t0 = time.time()
    # 默认 detector：face_model=retinaface, landmark_model=mobilefacenet,
    # au_model=xgb, emotion_model=resmasknet, identity_model=facenet
    detector = Detector(au_model="xgb")
    print(f"  ✅ Detector 加载完成 ({time.time()-t0:.1f}s)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头打不开")
        return
    print("\n  📷 摄像头打开成功，开始测试。请做：")
    print("     0-10s  自然脸（基线）")
    print("     10-20s 微笑")
    print("     20-30s 皱眉\n")

    au_history: list[dict] = []
    fc = 0
    last = 0.0
    t_start = time.time()
    try:
        while time.time() - t_start < 30:
            now = time.time()
            if now - last < 1.0:  # 1 秒一帧足够
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                continue
            last = now
            ret, frame = cap.read()
            if not ret:
                continue
            fc += 1

            # py-feat 0.6.2 用 detect_image(list of paths)，临时存盘再调
            try:
                tmp_path = f"/tmp/pyfeat_frame_{fc % 3}.jpg"
                cv2.imwrite(tmp_path, frame)
                fex = detector.detect_image([tmp_path], output_size=512)
                if fex is None or len(fex) == 0:
                    print(f"  #{fc}: 未检测到脸")
                    continue
                au_cols = [c for c in fex.columns if c.startswith("AU")]
                au_dict = {c: float(fex[c].iloc[0]) for c in au_cols}
                emo_cols = [c for c in ["anger","disgust","fear","happiness","sadness","surprise","neutral"] if c in fex.columns]
                emo = {c: float(fex[c].iloc[0]) for c in emo_cols}

                au_history.append({"frame": fc, "time": now - t_start, "au": au_dict, "emo": emo})

                # 简短输出
                au06 = au_dict.get("AU06", 0)
                au12 = au_dict.get("AU12", 0)
                au04 = au_dict.get("AU04", 0)
                au15 = au_dict.get("AU15", 0)
                au26 = au_dict.get("AU26", 0)
                top_emo = max(emo.items(), key=lambda x: x[1])
                print(f"  #{fc:>3} t={now-t_start:>5.1f}s | AU06={au06:.2f} AU12={au12:.2f} | AU04={au04:.2f} AU15={au15:.2f} AU26={au26:.2f} | top emotion: {top_emo[0]}({top_emo[1]:.0%})", flush=True)
            except Exception as e:
                print(f"  #{fc}: detect error: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not au_history:
        print("\n❌ 没有任何帧检测成功")
        return

    print(f"\n{'='*72}")
    print(f"📊 30 秒采样结果（{len(au_history)} 帧）— py-feat AU 实测分布")
    print(f"{'='*72}")
    print(f"  {'AU':<8} {'min':<8} {'max':<8} {'mean':<8} {'含义':<30}")
    au_meanings = {
        "AU01": "Inner Brow Raiser (惊讶/担忧)",
        "AU02": "Outer Brow Raiser",
        "AU04": "Brow Lowerer (皱眉)",
        "AU05": "Upper Lid Raiser (惊讶)",
        "AU06": "Cheek Raiser (Duchenne 笑)",
        "AU07": "Lid Tightener",
        "AU09": "Nose Wrinkler",
        "AU10": "Upper Lip Raiser",
        "AU11": "Nasolabial Deepener",
        "AU12": "Lip Corner Puller (微笑)",
        "AU14": "Dimpler",
        "AU15": "Lip Corner Depressor (悲伤)",
        "AU17": "Chin Raiser",
        "AU20": "Lip Stretcher",
        "AU23": "Lip Tightener",
        "AU24": "Lip Pressor",
        "AU25": "Lips Part",
        "AU26": "Jaw Drop",
        "AU28": "Lip Suck",
        "AU43": "Eyes Closed",
    }
    all_au = sorted({k for h in au_history for k in h["au"]})
    for au in all_au:
        vals = [h["au"].get(au, 0) for h in au_history]
        print(f"  {au:<8} {min(vals):<8.3f} {max(vals):<8.3f} {sum(vals)/len(vals):<8.3f} {au_meanings.get(au, '')}")

    print(f"\n  关键 AU 阶段对比（10s 划分）:")
    bands = [(0, 10, "自然脸"), (10, 20, "微笑"), (20, 30, "皱眉")]
    for t0, t1, label in bands:
        seg = [h for h in au_history if t0 <= h["time"] < t1]
        if not seg:
            continue
        au06 = sum(h["au"].get("AU06", 0) for h in seg) / len(seg)
        au12 = sum(h["au"].get("AU12", 0) for h in seg) / len(seg)
        au04 = sum(h["au"].get("AU04", 0) for h in seg) / len(seg)
        au15 = sum(h["au"].get("AU15", 0) for h in seg) / len(seg)
        print(f"    [{t0:>2}-{t1:>2}s] {label:<8} AU06={au06:.2f} AU12={au12:.2f} AU04={au04:.2f} AU15={au15:.2f}")

    print(f"\n  情绪分类（mean）:")
    all_emo = sorted({k for h in au_history for k in h["emo"]})
    for emo in all_emo:
        vals = [h["emo"].get(emo, 0) for h in au_history]
        print(f"    {emo:<14} mean={sum(vals)/len(vals):.2%}")


if __name__ == "__main__":
    main()
