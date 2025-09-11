import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


def create_gif_and_save(
    np_data: np.ndarray,
    path: str,
    vmin: float = 0.0,
    vmax: float = 40.0,
    fps: int = 2,
    bg_rgb=(255, 255, 255),  # 合成到白色背景，避免黑块2
):
    """
    np_data: (T, H, W) float 数组
    path: 输出 GIF 路径
    vmin, vmax: 颜色映射范围
    fps: 帧率
    bg_rgb: 背景颜色 (R,G,B)，用于对透明区域做 alpha 合成
    """
    # 1) 确保输出目录存在
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2) 准备数据与 colormap
    data = np.asarray(np_data, dtype=np.float32).copy()
    T, H, W = data.shape
    cmap = plt.get_cmap("viridis")

    # 3) 写入 GIF（注意 duration & disposal）
    duration = 1.0 / max(fps, 1)

    with imageio.get_writer(path, mode="I", loop=0) as writer:
        for t in range(T):
            arr = data[t]

            # 掩膜：<1 视为“无效/透明”
            mask = arr < 1.0

            # 归一化到 [0,1]
            if vmax == vmin:
                norm = np.zeros_like(arr, dtype=np.float32)
            else:
                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

            # 映射到 RGBA (H,W,4), uint8
            rgba = (cmap(norm) * 255).astype(np.uint8)

            # 将 mask 区域的 alpha 设为 0（完全透明）
            if mask.any():
                rgba[mask, 3] = 0

            # ---- 关键：先在内存里做一次 alpha 合成到固定背景 ----
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0  # (H,W,1)
            bg = np.zeros((H, W, 3), dtype=np.float32)
            bg[..., 0] = bg_rgb[0]
            bg[..., 1] = bg_rgb[1]
            bg[..., 2] = bg_rgb[2]

            rgb = (
                rgba[..., :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
            ).astype(np.uint8)

            # 以 RGB 方式写入，并设置每帧元信息
            # disposal=2 -> 播放器显示完这一帧后恢复背景，避免残影/延迟停留
            writer.append_data(rgb, meta={"duration": duration, "disposal": 2})