```markdown
# 图像频率分析：空域梯度法与FFT RMS法对比实验

## 实验目的

比较两种图像频率估计方法的一致性：
1. **空域梯度法**：基于图像梯度的快速频率估计
2. **FFT RMS法**：基于功率谱二阶矩的精确频率测量

## 实验原理

### 1. 空域梯度法
利用图像梯度与频率的关系，公式如下：
```
f_rms = sqrt( E[|∇I|²] / (4π² · Var(I)) )
```
其中：
- `|∇I|`：图像梯度幅值
- `Var(I)`：图像块方差
- `E[·]`：期望（均值）

### 2. FFT RMS法
通过傅里叶变换计算频率的加权平均：
```
f_rms = sqrt( ∑(f² · P(f)) / ∑P(f) )
```
其中：
- `f`：归一化频率
- `P(f)`：功率谱密度

## 实验环境

### 依赖库
```bash
pip install numpy matplotlib scipy scikit-image
```

## 代码结构

```
frequency_analysis/
├── main.py                    # 主程序
├── output/                    # 输出文件夹（自动创建）
│   ├── camera_gradient.png    # 空域梯度法频率图
│   ├── camera_fft_rms.png     # FFT RMS法频率图
│   ├── camera_comparison.png  # 综合对比图
│   ├── camera_gradient_freq.npy   # 梯度法数据
│   └── camera_fft_rms_freq.npy    # FFT法数据
└── README.md                  # 本文件
```

## 使用方法

### 1. 快速运行
```python
python main.py
```

### 2. 自定义图像
```python
from skimage import data
from main import FrequencyAnalyzer

# 使用自定义图像
img = data.coins()  # 或其他图像
analyzer = FrequencyAnalyzer(img)
results = analyzer.compare_gradient_vs_fft_rms()
```

### 3. 修改参数
在`FrequencyAnalyzer`类中调整：
```python
self.block_size = 16  # 修改块大小（16/32/64）
```

### 4. 可视化输出

#### 频率图（`camera_gradient.png` / `camera_fft_rms.png`）
- 热力图显示每个图像块的频率值
- 颜色越亮表示频率越高

#### 综合对比图（`camera_comparison.png`）
包含6个子图：
1. **原始图像**：输入图像
2. **空域梯度法频率图**：梯度法估计结果
3. **FFT RMS法频率图**：FFT法测量结果
4. **散点图**：两种方法的相关性（含y=x参考线）
5. **误差分布**：频率估计误差的直方图
6. **频率分布对比**：两种方法的频率分布

## 结果解读

### 一致性指标
- **相关系数**：越接近1表示一致性越好
  - 0.4-0.6：中等相关
  - 0.6-0.8：强相关
  - 0.8-1.0：极强相关

- **MAE/RMSE**：越小表示误差越小

### 系统性偏差
- **负偏差**：梯度法低估频率（常见于自然图像）
- **正偏差**：梯度法高估频率

### 典型结果（camera图像，16×16块）
| 指标 | 数值 | 解读 |
|------|------|------|
| 相关系数 | 0.39-0.44 | 中等相关，两种方法有一定一致性 |
| 梯度法均值 | ~0.30 | 梯度法估计的频率较低 |
| FFT法均值 | ~0.70 | FFT法测量的频率较高 |
| 偏差 | -0.40 | 梯度法系统性地低估频率 |

## 常见问题

### Q1: 为什么两种方法的结果差异较大？
**A**: 这是正常的，因为：
- 空域梯度法是近似方法，假设图像是单频信号
- 自然图像包含多种频率成分
- FFT方法是精确测量

### Q2: 梯度法为什么总是低估频率？
**A**: 可能原因：
- Sobel算子是梯度近似，会平滑高频
- 公式假设图像是正弦波，自然图像不满足
- 块内方差可能被低估
