import torch 
import torch.nn.functional as F
import torch.nn as nn

# --------1.utils--------
def make_grid(input):
    B, C, H, W = input.size()
    device, dtype = input.device, input.dtype
    # mesh grid
    xx = torch.arange(0, W, device=device, dtype=dtype).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device, dtype=dtype).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)   # [B,2,H,W]

    return grid


# 它根据 flow 指令，移动图像位置。
def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):

    B, C, H, W = input.size()
    vgrid = grid + flow

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1) # （B,H,W,2）
    # grid_sample 根据一个“网格”，从输入图像中采样像素值，返回一个变形后的新图像。
    output = F.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
    return output

# --------2.loss function--------

def w(x:torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0+x,max = 24.0)

def lwdis(x_t:torch.Tensor,x_hat_t : torch.Tensor) -> torch.Tensor:
    return torch.abs((x_t - x_hat_t)*w(x_t)).sum()

def j_accum(gt_frame,init_frame,intensity,motion,grid) -> torch.Tensor:
    '''
    gt_frame -> [B,20,H,W]
    init_frame -> [B,1,H,W]
    intensity -> [B,20,1,H,W]
    motion -> [B,20,2,H,W]
    grid -> [B,2,H,W]
    '''
    B,T,H,W = gt_frame.shape
    total = gt_frame.new_tensor(0.0)
    device = init_frame.device
    dtype  = init_frame.dtype
    prev = init_frame
    grid = grid.to(device=device, dtype=dtype)
    
    for t in range(T):
        x_t = gt_frame[:,t:t+1]  # [B,1,H,W]
        v_t = motion[:,t] # [B,2,H,W]
        s_t = intensity[:,t] # [B,1,H,W]
        
        x_t_prime_bili = warp(prev,v_t,grid,mode='bilinear')
        x_t_prime_near = warp(prev,v_t,grid,mode = 'nearest')
        x_t_double_prime = x_t_prime_near + s_t
        
        loss_residual = lwdis(x_t,x_t_double_prime)
        loss_motion = lwdis(x_t,x_t_prime_bili)
        total = total + loss_residual + loss_motion
        prev = x_t_double_prime.detach()
    
    return total/(B*T*H*W)


def sobel_filter(device="cuda", dtype=torch.float32):
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    return kx, ky


def j_motion(motion, gt_frame):
    """
    motion:   [B,T,2,H,W]
    gt_frame: [B,T,H,W]
    """
    B, T, C, H, W = motion.shape
    assert C == 2
    device, dtype = motion.device, motion.dtype

    flow = motion.reshape(B*T, C, H, W)                        # [B*T,2,H,W]
    weights = w(gt_frame).detach().reshape(B*T,1,H,W).to(device=device, dtype=dtype)

    kx, ky = sobel_filter(device=device, dtype=dtype)          # [1,1,3,3]
    kx_g = kx.repeat(C, 1, 1, 1)                               # [2,1,3,3]
    ky_g = ky.repeat(C, 1, 1, 1)                               # [2,1,3,3]

    gx = F.conv2d(flow, kx_g, padding=1, groups=C)             # [B*T,2,H,W]
    gy = F.conv2d(flow, ky_g, padding=1, groups=C)             # [B*T,2,H,W]
    grad_mag2 = gx.square() + gy.square()                      # [B*T,2,H,W]

    total = (grad_mag2 * weights).sum()                        # 空间+通道sum
    return total / (B*T*H*W)                                           # ★ 统一尺度
