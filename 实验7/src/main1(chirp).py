import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import os

# 创建output文件夹
if not os.path.exists('output'):
    os.makedirs('output')

# ============ 生成chirp测试图 ============
def generate_chirp(size=256, f_min=0.01, f_max=0.5):
    """生成chirp信号（频率递增的正弦波）"""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    # 径向频率递增
    r = np.sqrt((X-0.5)**2 + (Y-0.5)**2) * 2
    phase = 2 * np.pi * (f_min * r + (f_max - f_min) * r**2 / 2) * size / 2
    chirp = np.sin(phase)
    return chirp

def downsample(image, M):
    """M倍下采样"""
    return image[::M, ::M]

def compute_fft(image):
    """计算FFT频谱"""
    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log(np.abs(fft) + 1)
    return magnitude

def compute_psnr(original, reconstructed):
    """计算PSNR（峰值信噪比）"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(original) - np.min(original)
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

print("Chirp测试图抗混叠下采样实验（优化版）\n")
print("图片将保存到 output 文件夹\n")

# ============ 第一部分：Chirp测试图 ============
print("第一部分：基本抗混叠实验")

size = 256
chirp = generate_chirp(size, 0.01, 0.5)
M = 4

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(chirp, cmap='gray')
axes[0, 0].set_title(f'Original Chirp Signal ({size}x{size})')
axes[0, 0].axis('off')

# 直接下采样
downsampled_direct = downsample(chirp, M)
axes[0, 1].imshow(downsampled_direct, cmap='gray')
axes[0, 1].set_title(f'Direct {M}x Downsampling (Aliasing)')
axes[0, 1].axis('off')

# 高斯滤波后再下采样
sigma = 0.45 * M
filtered = gaussian_filter(chirp, sigma=sigma)
downsampled_filtered = downsample(filtered, M)
axes[0, 2].imshow(downsampled_filtered, cmap='gray')
axes[0, 2].set_title(f'Gaussian Filter (sigma={sigma:.2f}) + Downsampling')
axes[0, 2].axis('off')

# FFT频谱
fft_original = compute_fft(chirp)
fft_direct = compute_fft(downsampled_direct)
fft_filtered = compute_fft(downsampled_filtered)

axes[1, 0].imshow(fft_original, cmap='hot')
axes[1, 0].set_title('Original FFT Spectrum')
axes[1, 0].axis('off')

axes[1, 1].imshow(fft_direct, cmap='hot')
axes[1, 1].set_title('Direct Downsampling FFT (Aliasing)')
axes[1, 1].axis('off')

axes[1, 2].imshow(fft_filtered, cmap='hot')
axes[1, 2].set_title('Filtered Downsampling FFT (No Aliasing)')
axes[1, 2].axis('off')

plt.suptitle('Part 1: Chirp Signal Anti-aliasing Downsampling Experiment', fontsize=14)
plt.tight_layout()
plt.savefig('output/part1_chirp_antialiasing.png', dpi=150, bbox_inches='tight')
plt.show()
print("已保存: output/part1_chirp_antialiasing.png")

# ============ 第二部分：验证σ公式 ============
print("\n第二部分：验证σ公式")

M = 4
theoretical_sigma = 0.45 * M

chirp = generate_chirp(256, 0.01, 0.5)

# 创建2行4列的子图来显示所有结果
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

# 原图
axes[0, 0].imshow(chirp, cmap='gray')
axes[0, 0].set_title('Original Chirp Signal')
axes[0, 0].axis('off')

# 直接下采样
direct_down = downsample(chirp, M)
axes[0, 1].imshow(direct_down, cmap='gray')
axes[0, 1].set_title(f'Direct {M}x Downsampling\n(Aliasing)')
axes[0, 1].axis('off')

# 理论最优值
filtered_theory = gaussian_filter(chirp, sigma=theoretical_sigma)
theory_down = downsample(filtered_theory, M)
axes[0, 2].imshow(theory_down, cmap='gray')
axes[0, 2].set_title(f'Theoretical sigma={theoretical_sigma}\n(Optimal)')
axes[0, 2].axis('off')

# 显示不同的sigma值
sigma_small = [0.5, 1.0]
for idx, sigma in enumerate(sigma_small):
    filtered = gaussian_filter(chirp, sigma=sigma)
    downsampled = downsample(filtered, M)
    axes[1, idx].imshow(downsampled, cmap='gray')
    axes[1, idx].set_title(f'sigma={sigma}\n(sigma too small, aliasing remains)')
    axes[1, idx].axis('off')

sigma_large = [2.5, 4.0]
for idx, sigma in enumerate(sigma_large):
    filtered = gaussian_filter(chirp, sigma=sigma)
    downsampled = downsample(filtered, M)
    axes[1, idx+2].imshow(downsampled, cmap='gray')
    axes[1, idx+2].set_title(f'sigma={sigma}\n(sigma too large, over-blurring)')
    axes[1, idx+2].axis('off')

plt.suptitle(f'Part 2: Effect of Different sigma Values on {M}x Downsampling (Chirp Signal)', fontsize=14)
plt.tight_layout()
plt.savefig('output/part2_chirp_sigma_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"已保存: output/part2_chirp_sigma_comparison.png")
print(f"\n理论σ值: σ = 0.45 × {M} = {theoretical_sigma}")
print("结论：σ=1.8附近效果最佳，混叠消失且不过度模糊")

# ============ 第三部分：自适应下采样（优化版） ============
print("\n第三部分：自适应下采样（优化版）")

def estimate_local_M_optimized(image, block_size=8, max_M=8):
    """优化的局部M值估计 - 使用局部频率分析"""
    h, w = image.shape
    M_map = np.zeros((h // block_size, w // block_size))
    
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size]
            
            # 计算局部梯度（更简单直接）
            gy, gx = np.gradient(block)
            grad_mag = np.sqrt(gx**2 + gy**2)
            
            # 计算梯度的方差（反映纹理复杂度）
            grad_std = np.std(grad_mag)
            grad_mean = np.mean(grad_mag)
            
            # 综合指标：梯度强度和变化
            complexity = grad_mean + 0.5 * grad_std
            
            # 根据复杂度映射到M值（复杂度越高，M越小）
            if complexity < 0.05:
                M_est = max_M  # 平坦区域，可以用大M
            elif complexity < 0.15:
                M_est = int(max_M * 0.75)
            elif complexity < 0.3:
                M_est = int(max_M * 0.5)
            else:
                M_est = 2  # 复杂区域，用小M
            
            M_est = max(2, min(max_M, M_est))
            
            map_i = i // block_size
            map_j = j // block_size
            M_map[map_i, map_j] = M_est
    
    # 中值滤波平滑M值
    from scipy.ndimage import median_filter
    M_map_smooth = median_filter(M_map, size=3)
    
    return M_map_smooth.astype(int)

def adaptive_downsample_optimized(image, target_size):
    """优化的自适应下采样"""
    h, w = image.shape
    target_h, target_w = target_size
    
    # 使用优化的M值估计
    M_map = estimate_local_M_optimized(image)
    
    # 创建结果图像
    result = np.zeros(target_size)
    
    block_h = h // M_map.shape[0]
    block_w = w // M_map.shape[1]
    
    for i in range(M_map.shape[0]):
        for j in range(M_map.shape[1]):
            M = int(M_map[i, j])
            y_start = i * block_h
            y_end = min((i+1) * block_h, h)
            x_start = j * block_w
            x_end = min((j+1) * block_w, w)
            
            block = image[y_start:y_end, x_start:x_end]
            
            # 根据M值自适应选择sigma
            sigma = 0.45 * M
            filtered = gaussian_filter(block, sigma=sigma)
            
            # 下采样
            downsampled = filtered[::M, ::M]
            
            # 计算在目标图像中的位置
            target_y_start = int(i * target_h / M_map.shape[0])
            target_y_end = int((i+1) * target_h / M_map.shape[0])
            target_x_start = int(j * target_w / M_map.shape[1])
            target_x_end = int((j+1) * target_w / M_map.shape[1])
            
            # 调整大小并放入结果
            downsampled_resized = resize(downsampled, 
                                        (target_y_end-target_y_start, 
                                         target_x_end-target_x_start),
                                        preserve_range=True)
            result[target_y_start:target_y_end, 
                   target_x_start:target_x_end] = downsampled_resized
    
    return result

size = 256
# 创建多尺度chirp测试图
chirp_varied = np.zeros((size, size))
# 左上角：低频
chirp_varied[:128, :128] = generate_chirp(128, 0.01, 0.12)
# 右上角：中低频
chirp_varied[:128, 128:] = generate_chirp(128, 0.08, 0.28)
# 左下角：中高频
chirp_varied[128:, :128] = generate_chirp(128, 0.12, 0.35)
# 右下角：高频
chirp_varied[128:, 128:] = generate_chirp(128, 0.2, 0.6)

M = 4
target_size = (size // M, size // M)

# 方法1：直接下采样
direct_down = downsample(chirp_varied, M)

# 方法2：统一滤波后下采样
uniform_sigma = 0.45 * M
uniform_filtered = gaussian_filter(chirp_varied, sigma=uniform_sigma)
uniform_down = downsample(uniform_filtered, M)

# 方法3：优化的自适应下采样
adaptive_down = adaptive_downsample_optimized(chirp_varied, target_size)

# 计算误差和PSNR
direct_upsampled = resize(direct_down, chirp_varied.shape, preserve_range=True)
uniform_upsampled = resize(uniform_down, chirp_varied.shape, preserve_range=True)
adaptive_upsampled = resize(adaptive_down, chirp_varied.shape, preserve_range=True)

error_direct = np.abs(chirp_varied - direct_upsampled)
error_uniform = np.abs(chirp_varied - uniform_upsampled)
error_adaptive = np.abs(chirp_varied - adaptive_upsampled)

mse_direct = np.mean(error_direct**2)
mse_uniform = np.mean(error_uniform**2)
mse_adaptive = np.mean(error_adaptive**2)

psnr_direct = compute_psnr(chirp_varied, direct_upsampled)
psnr_uniform = compute_psnr(chirp_varied, uniform_upsampled)
psnr_adaptive = compute_psnr(chirp_varied, adaptive_upsampled)

# 显示结果
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

axes[0, 0].imshow(chirp_varied, cmap='gray')
axes[0, 0].set_title('Original Multi-scale Chirp Signal')
axes[0, 0].axis('off')

axes[0, 1].imshow(direct_down, cmap='gray')
axes[0, 1].set_title('Direct Downsampling')
axes[0, 1].axis('off')

axes[0, 2].imshow(uniform_down, cmap='gray')
axes[0, 2].set_title('Uniform Filtering + Downsampling')
axes[0, 2].axis('off')

axes[0, 3].imshow(adaptive_down, cmap='gray')
axes[0, 3].set_title('Adaptive Downsampling (Optimized)')
axes[0, 3].axis('off')

# 误差图
im1 = axes[1, 0].imshow(error_direct, cmap='hot')
axes[1, 0].set_title(f'Direct Error\nMSE: {mse_direct:.4f}, PSNR: {psnr_direct:.1f}dB')
axes[1, 0].axis('off')
plt.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].imshow(error_uniform, cmap='hot')
axes[1, 1].set_title(f'Uniform Error\nMSE: {mse_uniform:.4f}, PSNR: {psnr_uniform:.1f}dB')
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1])

im3 = axes[1, 2].imshow(error_adaptive, cmap='hot')
axes[1, 2].set_title(f'Adaptive Error\nMSE: {mse_adaptive:.4f}, PSNR: {psnr_adaptive:.1f}dB')
axes[1, 2].axis('off')
plt.colorbar(im3, ax=axes[1, 2])

# M值分布图
M_map = estimate_local_M_optimized(chirp_varied)
im4 = axes[1, 3].imshow(M_map, cmap='viridis', vmin=2, vmax=8)
axes[1, 3].set_title('M Value Distribution\n(Dark: small M, Bright: large M)')
axes[1, 3].axis('off')
plt.colorbar(im4, ax=axes[1, 3])

plt.suptitle('Part 3: Adaptive Downsampling Comparison (Chirp Signal - Optimized)', fontsize=14)
plt.tight_layout()
plt.savefig('output/part3_chirp_adaptive_optimized.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"已保存: output/part3_chirp_adaptive_optimized.png")
print(f"\n========== 定量评估结果 ==========")
print(f"直接下采样     - MSE: {mse_direct:.6f}, PSNR: {psnr_direct:.2f} dB")
print(f"统一滤波下采样 - MSE: {mse_uniform:.6f}, PSNR: {psnr_uniform:.2f} dB")
print(f"自适应下采样   - MSE: {mse_adaptive:.6f}, PSNR: {psnr_adaptive:.2f} dB")

if mse_adaptive < mse_uniform:
    improvement_mse = (mse_uniform - mse_adaptive) / mse_uniform * 100
    improvement_psnr = psnr_adaptive - psnr_uniform
    print(f"\n✓ 自适应方法优于统一滤波:")
    print(f"  MSE降低 {improvement_mse:.2f}%, PSNR提升 {improvement_psnr:.2f} dB")
else:
    print(f"\n⚠ 自适应方法效果略逊于统一滤波")
    print(f"  这是chirp信号频率连续变化的特性导致的")
    print(f"  对于频率平滑变化的信号，统一滤波可能更合适")

# 显示M值分布
fig, ax = plt.subplots(figsize=(8, 6))
unique, counts = np.unique(M_map, return_counts=True)
ax.bar(unique, counts, width=0.6, alpha=0.7, edgecolor='black', color='steelblue')
ax.set_xlabel('M Value', fontsize=12)
ax.set_ylabel('Number of Blocks', fontsize=12)
ax.set_title('Distribution of Adaptive M Values', fontsize=14)
ax.set_xticks(unique)
ax.grid(True, alpha=0.3)
for i, (u, c) in enumerate(zip(unique, counts)):
    ax.text(u, c + 0.5, str(c), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('output/m_value_distribution_optimized.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"已保存: output/m_value_distribution_optimized.png")

print("\n所有实验完成！图片已保存到 output 文件夹")
print("\n优化说明：")
print("1. 使用梯度方差+均值综合评估纹理复杂度")
print("2. 采用离散的M值映射策略（2,4,6,8）")
print("3. 使用中值滤波平滑M值分布")
print("4. 添加PSNR作为客观评价指标")
print("5. 针对chirp信号特点调整了频率范围划分")