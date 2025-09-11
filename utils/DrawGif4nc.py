import xarray as xr
import numpy as np
import imageio, os, glob
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from concurrent.futures import ProcessPoolExecutor, as_completed

def drawnc(path, fps, vmin, vmax, mask_thr, var):
    """
    处理单个 .nc 文件并生成 GIF。

    Args:
        path (str): .nc 文件的路径。
        fps (int): GIF 的帧率。
        vmin (float): 绘图的颜色最小值。
        vmax (float): 绘图的颜色最大值。
        mask_thr (float): 小于此值的降雨量将被置为 NaN。
        var (str): 数据变量名。
    """
    # 获取文件名，并根据 fps 参数生成唯一的 GIF 路径
    name = os.path.splitext(os.path.basename(path))[0]
    # 在文件名中加入 fps 信息，避免不同 fps 任务间的输出文件冲突
    gif_path = os.path.join(r"C:\NowcastNet\graph", f"{name}_fps{fps}.gif")

    # 读取数据
    ds = xr.open_dataset(path)
    da = ds[var]
    lon2d = ds["longitude"].values
    lat2d = ds["latitude"].values

    proj = ccrs.PlateCarree()

    # 在这个 for 循环中，由于 ProcessPoolExecutor 会自动创建新进程，
    # 这里的 tqdm 是不需要的，否则会创建多个进度条。
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for t in da.time.values:
            arr = da.sel(time=t).values.astype(np.float32)
            arr[arr < mask_thr] = np.nan

            fig, ax = plt.subplots(figsize=(6, 5), dpi=150, subplot_kw={"projection": proj})
            ax.coastlines(resolution="10m", color="black", linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)

            pm = ax.pcolormesh(
                lon2d,
                lat2d,
                arr,
                transform=proj,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
            )
            ax.set_extent(
                [np.nanmin(lon2d), np.nanmax(lon2d), np.nanmin(lat2d), np.nanmax(lat2d)],
                crs=proj,
            )

            # 修正：标签更严谨地反映了数据的物理单位
            # 假设数据单位是毫米/小时，这是气象领域常见单位
            cbar = plt.colorbar(
                pm, ax=ax, fraction=0.046, pad=0.04, label="Rainfall (mm)"
            )
            # 修正：标题更明确地表示时间点
            ax.set_title(np.datetime_as_string(t, unit="m").replace('T', ' '))

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            rgba = buf[:, :, [1, 2, 3, 0]]
            writer.append_data(rgba)
            plt.close(fig)
            
    # 返回有用的信息，方便主进程追踪
    return f"GIF 已生成：{name}"

if __name__ == '__main__':
    fps = 5
    vmin, vmax = 0.0, 40.0
    mask_thr = 1.0
    var = "rainfall"

    paths = glob.glob(r'D:\NowcastNet\XRAIN_10min_nc\*.nc')
    
    # 使用 ProcessPoolExecutor 来并行执行任务
    with ProcessPoolExecutor(max_workers=None) as executor:
        # 提交所有任务到进程池，并建立 future 到任务参数的映射
        futures = {executor.submit(drawnc, path, fps, vmin, vmax, mask_thr, var): path for path in paths}

        # 等待任务完成，并显示总进度
        for future in tqdm(as_completed(futures), total=len(paths), desc="处理文件"):
            try:
                # 获取任务结果，如果任务失败会抛出异常
                future.result()
            except Exception as e:
                task_info = futures[future]
                # 打印详细的错误信息
                print(f"任务 {task_info[0]} (fps={task_info[1]}) 处出错了：{e}")

        print("所有文件处理完成！")