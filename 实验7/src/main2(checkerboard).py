import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import os

# 创建output1文件夹
if not os.path.exists('output1'):
    os.makedirs('output1')

# ============ 生成棋盘格 ============
def generate_checkerboard(size=256, square_size=8):
    """生成棋盘格"""
    n_squares = size // square_size
    checker = np.zeros((size, size))
    for i in range(n_squares):
        for j in range(n_squares):
            if (i + j) % 2 == 0:
                checker[i*square_size:(i+1)*square_size, 
                       j*square_size:(j+1)*square_size] = 1
    return checker

def downsample(image, M):
    """M倍下采样"""
    return image[::M, ::M]

def compute_fft(image):
    """计算FFT频谱"""
    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log(np.abs(fft) + 1)
    return magnitude

print("棋盘格抗混叠下采样实验\n")
print("图片将保存到 output1 文件夹\n")

# ============ 第一部分：棋盘格测试 ============
print("第一部分：基本抗混叠实验")

size = 256
square_size = 8
checker = generate_checkerboard(size, square_size)
M = 4

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(checker, cmap='gray')
axes[0, 0].set_title(f'Original Checkerboard ({size}x{size})')
axes[0, 0].axis('off')

# 直接下采样
downsampled_direct = downsample(checker, M)
axes[0, 1].imshow(downsampled_direct, cmap='gray')
axes[0, 1].set_title(f'Direct {M}x Downsampling (Aliasing)')
axes[0, 1].axis('off')

# 高斯滤波后再下采样
sigma = 0.45 * M
filtered = gaussian_filter(checker, sigma=sigma)
downsampled_filtered = downsample(filtered, M)
axes[0, 2].imshow(downsampled_filtered, cmap='gray')
axes[0, 2].set_title(f'Gaussian Filter (sigma={sigma:.2f}) + Downsampling')
axes[0, 2].axis('off')

# FFT频谱
fft_original = compute_fft(checker)
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

plt.suptitle('Part 1: Checkerboard Anti-aliasing Downsampling Experiment', fontsize=14)
plt.tight_layout()
plt.savefig('output1/part1_checkerboard_antialiasing.png', dpi=150, bbox_inches='tight')
plt.show()
print("已保存: output1/part1_checkerboard_antialiasing.png")

# ============ 第二部分：验证σ公式 ============
print("\n第二部分：验证σ公式")

M = 4
theoretical_sigma = 0.45 * M

checker = generate_checkerboard(256, 8)

# 创建2行4列的子图来显示所有结果
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

# 原图
axes[0, 0].imshow(checker, cmap='gray')
axes[0, 0].set_title('Original Checkerboard')
axes[0, 0].axis('off')

# 直接下采样
direct_down = downsample(checker, M)
axes[0, 1].imshow(direct_down, cmap='gray')
axes[0, 1].set_title(f'Direct {M}x Downsampling\n(Aliasing)')
axes[0, 1].axis('off')

# 理论最优值
filtered_theory = gaussian_filter(checker, sigma=theoretical_sigma)
theory_down = downsample(filtered_theory, M)
axes[0, 2].imshow(theory_down, cmap='gray')
axes[0, 2].set_title(f'Theoretical sigma={theoretical_sigma}\n(Optimal)')
axes[0, 2].axis('off')

# 显示不同的sigma值（小于理论值）
sigma_small = [0.5, 1.0]
for idx, sigma in enumerate(sigma_small):
    filtered = gaussian_filter(checker, sigma=sigma)
    downsampled = downsample(filtered, M)
    axes[1, idx].imshow(downsampled, cmap='gray')
    axes[1, idx].set_title(f'sigma={sigma}\n(sigma too small, aliasing remains)')
    axes[1, idx].axis('off')

# 显示不同的sigma值（大于理论值）
sigma_large = [2.5, 4.0]
for idx, sigma in enumerate(sigma_large):
    filtered = gaussian_filter(checker, sigma=sigma)
    downsampled = downsample(filtered, M)
    axes[1, idx+2].imshow(downsampled, cmap='gray')
    axes[1, idx+2].set_title(f'sigma={sigma}\n(sigma too large, over-blurring)')
    axes[1, idx+2].axis('off')

plt.suptitle(f'Part 2: Effect of Different sigma Values on {M}x Downsampling (Checkerboard)', fontsize=14)
plt.tight_layout()
plt.savefig('output1/part2_checkerboard_sigma_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"已保存: output1/part2_checkerboard_sigma_comparison.png")
print(f"\n理论σ值: σ = 0.45 × {M} = {theoretical_sigma}")
print("结论：σ=1.8附近效果最佳，混叠消失且不过度模糊")

# ============ 第三部分：自适应下采样 ============
print("\n第三部分：自适应下采样")

def estimate_local_M(image, block_size=16, max_M=8):
    """基于梯度估计局部M值"""
    h, w = image.shape
    M_map = np.ones((h // block_size, w // block_size)) * max_M
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            
            # 计算梯度
            grad_y, grad_x = np.gradient(block)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            mean_grad = np.mean(grad_mag)
            
            # 梯度大的地方用小M，梯度小的地方用大M
            M_est = int(max_M * (1 - np.clip(mean_grad / 30, 0, 0.7)))
            M_est = max(2, min(max_M, M_est))
            
            M_map[i//block_size, j//block_size] = M_est
    
    return M_map

def adaptive_downsample(image, target_size):
    """自适应下采样"""
    h, w = image.shape
    target_h, target_w = target_size
    
    M_map = estimate_local_M(image)
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
            sigma = 0.45 * M
            filtered = gaussian_filter(block, sigma=sigma)
            downsampled = filtered[::M, ::M]
            
            # 调整到目标大小
            target_y_start = int(i * target_h / M_map.shape[0])
            target_y_end = int((i+1) * target_h / M_map.shape[0])
            target_x_start = int(j * target_w / M_map.shape[1])
            target_x_end = int((j+1) * target_w / M_map.shape[1])
            
            downsampled_resized = resize(downsampled, 
                                        (target_y_end-target_y_start, 
                                         target_x_end-target_x_start))
            result[target_y_start:target_y_end, 
                   target_x_start:target_x_end] = downsampled_resized
    
    return result

size = 256
# 创建多尺度棋盘格
checker_varied = np.zeros((size, size))
checker_varied[:128, :128] = generate_checkerboard(128, 4)   # 高频区
checker_varied[128:, 128:] = generate_checkerboard(128, 16)  # 低频区
checker_varied[:128, 128:] = generate_checkerboard(128, 8)   # 中频区
checker_varied[128:, :128] = generate_checkerboard(128, 8)   # 中频区

M = 4
target_size = (size // M, size // M)

# 方法1：直接下采样
direct_down = downsample(checker_varied, M)

# 方法2：统一滤波后下采样
uniform_sigma = 0.45 * M
uniform_filtered = gaussian_filter(checker_varied, sigma=uniform_sigma)
uniform_down = downsample(uniform_filtered, M)

# 方法3：自适应下采样
adaptive_down = adaptive_downsample(checker_varied, target_size)

# 计算误差
direct_upsampled = resize(direct_down, checker_varied.shape)
uniform_upsampled = resize(uniform_down, checker_varied.shape)
adaptive_upsampled = resize(adaptive_down, checker_varied.shape)

error_direct = np.abs(checker_varied - direct_upsampled)
error_uniform = np.abs(checker_varied - uniform_upsampled)
error_adaptive = np.abs(checker_varied - adaptive_upsampled)

# 显示结果
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

axes[0, 0].imshow(checker_varied, cmap='gray')
axes[0, 0].set_title('Original Multi-scale Checkerboard')
axes[0, 0].axis('off')

axes[0, 1].imshow(direct_down, cmap='gray')
axes[0, 1].set_title('Direct Downsampling')
axes[0, 1].axis('off')

axes[0, 2].imshow(uniform_down, cmap='gray')
axes[0, 2].set_title('Uniform Filtering + Downsampling')
axes[0, 2].axis('off')

axes[0, 3].imshow(adaptive_down, cmap='gray')
axes[0, 3].set_title('Adaptive Downsampling')
axes[0, 3].axis('off')

# 误差图
im1 = axes[1, 0].imshow(error_direct, cmap='hot')
axes[1, 0].set_title(f'Direct Error: {np.mean(error_direct):.4f}')
axes[1, 0].axis('off')
plt.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].imshow(error_uniform, cmap='hot')
axes[1, 1].set_title(f'Uniform Filter Error: {np.mean(error_uniform):.4f}')
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1])

im3 = axes[1, 2].imshow(error_adaptive, cmap='hot')
axes[1, 2].set_title(f'Adaptive Filter Error: {np.mean(error_adaptive):.4f}')
axes[1, 2].axis('off')
plt.colorbar(im3, ax=axes[1, 2])

# M值分布图
M_map = estimate_local_M(checker_varied)
im4 = axes[1, 3].imshow(M_map, cmap='viridis')
axes[1, 3].set_title('M Value Distribution (Dark: small M, Bright: large M)')
axes[1, 3].axis('off')
plt.colorbar(im4, ax=axes[1, 3])

plt.suptitle('Part 3: Adaptive Downsampling Comparison Experiment (Checkerboard)', fontsize=14)
plt.tight_layout()
plt.savefig('output1/part3_checkerboard_adaptive_downsample.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"已保存: output1/part3_checkerboard_adaptive_downsample.png")
print(f"\n直接下采样平均误差: {np.mean(error_direct):.6f}")
print(f"统一滤波下采样平均误差: {np.mean(error_uniform):.6f}")
print(f"自适应下采样平均误差: {np.mean(error_adaptive):.6f}")
improvement = (np.mean(error_uniform) - np.mean(error_adaptive)) / np.mean(error_uniform) * 100
print(f"自适应方法改进: {improvement:.2f}%")

print("\n所有实验完成！图片已保存到 output1 文件夹")