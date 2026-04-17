# Fixed Version 差异分析报告

## 概述
本次修复主要解决了**体素采样位置**的问题：将采样点从**网格节点 (grid nodes)** 改为**体素中心 (cell centers)**。

---

## 文件变更详情

### 1. `src/pfcsdf/geometry/volume.py` - 核心体素网格修改

**新增属性:**
- `cell_center_origin`: 计算体素中心的起始坐标 `origin + 0.5 * spacing`
- `x_cell_centers`, `y_cell_centers`, `z_cell_centers`: 各轴向上的体素中心坐标数组

**新增方法:**
- `cell_center_point(i, j, k)`: 获取指定索引的体素中心坐标
- `cell_center_mesh_coordinates()`: 生成所有体素中心的网格坐标
- `stacked_cell_centers()`: 返回所有体素中心坐标的堆叠数组 (shape: `(ni, nj, nk, 3)`)

**关键差异:**
```python
# 原版 (pfc_sdf_validation)
# - 只有 point(), stacked_points() 等方法
# - 采样点 = 网格节点 (grid nodes/vertices)

# 修复版 (fixed)
# - 新增 cell_center_* 系列方法
# - 采样点 = 体素中心 (cell centers)
```

---

### 2. `src/pfcsdf/contact/native_band.py` - SDF 梯度计算和采样修改

**新增函数:**
```python
def _sdf_gradient(sdf: Any, point: ArrayLike, eps: float = 1e-6) -> ArrayLike:
    """计算 SDF 梯度: 优先调用 sdf.gradient(), 否则使用中心有限差分"""
    if hasattr(sdf, "gradient"):
        return np.asarray(sdf.gradient(point), dtype=float)
    
    # 有限差分降级方案
    grad = np.zeros(3, dtype=float)
    for axis in range(3):
        step = np.zeros(3, dtype=float)
        step[axis] = eps
        plus = float(sdf.signed_distance(point + step))
        minus = float(sdf.signed_distance(point - step))
        grad[axis] = (plus - minus) / (2.0 * eps)
    return grad
```

**修改导入:**
- 移除: `from pfcsdf.geometry.base import SignedDistanceGeometry, signed_distance_gradient`
- 新增: `from typing import Any`

**函数签名修改:**
```python
# 原版
def sample_linear_pfc_balance_fields(
    grid: UniformGrid3D,
    sdf_a: SignedDistanceGeometry,  # 强类型约束
    sdf_b: SignedDistanceGeometry,
    ...

# 修复版
def sample_linear_pfc_balance_fields(
    grid: UniformGrid3D,
    sdf_a: Any,  # 弱类型, 支持更多 SDF 实现
    sdf_b: Any,
    ...
```

**采样点修改 (核心修复):**
```python
# 原版 - 使用网格节点
x = grid.point(i, j, k)
grad_phi_a = signed_distance_gradient(sdf_a, x)

# 修复版 - 使用体素中心
x = grid.cell_center_point(i, j, k)
grad_phi_a = _sdf_gradient(sdf_a, x)
```

**其他调用点修改:**
- `accumulate_sdf_native_band_wrench`: `stacked_points()` → `stacked_cell_centers()`
- `build_sparse_active_traversal`: `stacked_points()` → `stacked_cell_centers()`
- `_linearized_sample_at_offset`: `grid.point()` → `grid.cell_center_point()`

---

### 3. `src/pfcsdf/dynamics/benchmarks.py` - Benchmark 模型修改

**NativeBandSphereContactModel.__init__:**
```python
# 原版
points = self.grid.stacked_cell_centers()  # 注意: 原版 volume.py 没有此方法!

# 修复版
points = self.grid.stacked_cell_centers()  # 现在 volume.py 有此方法
```

**NativeBandFlatContactModel.__init__:**
```python
# 原版
points = self.grid.stacked_cell_centers()

# 修复版
points = self.grid.stacked_cell_centers()
```

> **注意**: 原版代码中 benchmarks.py 调用了 `stacked_cell_centers()` 但 volume.py 并未定义此方法,
> 这可能是原版的笔误或实际运行时有其他实现。修复版在 volume.py 中正确定义了这个方法。

---

### 4. `src/pfcsdf/experiments/ablation.py` - 网格参数调整

**网格分辨率调整:**
```python
# 原版
grid = UniformGrid3D(
    origin=np.array([-0.4, -0.4, -0.1]),
    spacing=np.array([0.2, 0.2, 0.01]),
    shape=(5, 5, 21),  # 5x5x21 网格
)

# 修复版
grid = UniformGrid3D(
    origin=np.array([-0.4, -0.4, -0.1]),
    spacing=np.array([0.2, 0.2, 0.01]),
    shape=(4, 4, 20),  # 4x4x20 网格 (各维度减 1)
)
```

**原因分析:**
- 当采样从网格节点改为体素中心时, 相同物理区域内的采样点数会减少
- `(5, 5, 21)` 节点网格 → `(4, 4, 20)` 体素中心网格 是合理的对应关系

---

## 修复影响总结

| 维度 | 原版 (pfc_sdf_validation) | 修复版 (fixed) |
|------|--------------------------|----------------|
| **采样位置** | 网格节点 (grid nodes) | 体素中心 (cell centers) |
| **梯度计算** | 专用 `signed_distance_gradient()` | 通用 `_sdf_gradient()` (支持有限差分) |
| **类型约束** | `SignedDistanceGeometry` 接口 | `Any` 类型 (duck typing) |
| **物理意义** | 在网格顶点处采样 | 在体素单元中心采样 |
| **网格分辨率** | shape=(5, 5, 21) | shape=(4, 4, 20) |

---

## 为什么这个修复是重要的?

1. **物理正确性**: 体素 (voxel) 本质上是一个空间区域, 其代表性采样点应该是中心而非顶点
2. **数值积分精度**: 在体素中心采样符合有限体积法的标准实践
3. **SDF 梯度一致性**: 体素中心处的 SDF 梯度更准确地反映了局部几何特征
4. **接触力计算**: 接触力和力矩在体素中心处的积分更为准确

---

## 生成时间
2026-04-17
