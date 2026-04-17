# Native-band 体素 / SDF 采样修复迁移说明

这份文档用于指导本地 `codex` 将最新修复从 `fixed/` 目录迁移到你的当前项目分支。

## 1. 这次修复要解决的核心问题

### 现象
在修复前，`native-band flat` 相对解析参考的曲线和指标明显偏差过大，典型表现为：

- 峰值力偏高；
- 力脉冲更尖、更窄；
- 接触结束时间明显提前；
- 与 analytic flat exact 的 reference comparison 差异过大。

### 根因
根因不是单纯的“z 向体素分辨率太粗”，而是 **native-band 的体素几何语义不一致**：

1. `UniformGrid3D.point(i, j, k)` 返回的是 **格点 / cell lower corner 语义**；
2. 但 native-band 的 SDF 采样、active mask 和 wrench 累加逻辑又把这些点当成了 **cell center** 在使用；
3. `footprint.contains_xy(...)` 又按“采样点是否落在 footprint 内”来判定有效接触区域；
4. 结果导致 xy 平面的有效支撑面积被系统性放大，进而把 force / impulse / release timing 一起带偏。

一句话总结：

> 旧实现的问题不是 generic voxel resolution 不足，而是 **grid point 与 cell center 混用**，导致 native-band 的体积采样和 footprint support mask 出现了系统性偏差。

---

## 2. 这次修改的总目标

这次修改的目标不是“微调参数”，而是把 native-band 路径统一成明确的 **cell-centered volumetric sampling** 语义。

最终要达到的定义是：

> **native-band now uses cell-centered volumetric sampling and cell-centered footprint support masking**

也就是说：

- SDF / balance field 的体积采样按 **cell center** 做；
- active support / footprint mask 的判定按 **cell center** 做；
- wrench accumulation 的作用点按 **cell center** 做；
- 配置中的 `origin` 继续表示 **grid bounds 的 lower corner**，而不是“第一个 cell center”。

---

## 3. 迁移时的源码真值来源

你已经说明会把最新版文件放到一个 `fixed/` 文件夹中。迁移时建议把 `fixed/` 视为唯一真值来源。

推荐迁移顺序：

1. `fixed/src/pfcsdf/geometry/volume.py`
2. `fixed/src/pfcsdf/contact/native_band.py`
3. `fixed/src/pfcsdf/dynamics/benchmarks.py`
4. `fixed/configs/main_tables.yaml`
5. `fixed/configs/efficiency.yaml`
6. `fixed/src/pfcsdf/experiments/ablation.py`
7. `fixed/tests/...`
8. `fixed/pfc_sdf_dynamics_paper_sprint1.tex` / `fixed/pfc_sdf_dynamics_paper_sprint3_appendix.tex`（如果你也想同步论文描述）

---

## 4. 具体修改了哪些地方，以及每处修改的目标是什么

### 4.1 `src/pfcsdf/geometry/volume.py`

#### 修改内容
在 `UniformGrid3D` 中新增了明确的 **cell-center API**：

- `cell_center_origin`
- `x_cell_centers`
- `y_cell_centers`
- `z_cell_centers`
- `cell_center_point(i, j, k)`
- `cell_center_mesh_coordinates()`
- `stacked_cell_centers()`

#### 修改目标
把“格点语义”和“cell center 语义”显式分开，避免再靠隐式约定使用 `point()`。

#### 迁移要点
- **不要删除** 旧的 `point()` 和 `stacked_points()`；
- 旧 API 继续保留为 corner / grid-point 语义；
- 新 API 专门提供给 native-band 这类需要 cell-centered 体积采样的路径使用。

#### 这一步解决什么问题
它把“采样位置到底是 corner 还是 center”从隐式假设变成了显式接口，是整个修复的语义基础。

---

### 4.2 `src/pfcsdf/contact/native_band.py`

#### 修改内容
把 native-band 的关键采样和累加逻辑统一切换到 **cell center**：

1. 采样 SDF / depth / balance field 时：
   - 从 `grid.point(i, j, k)` 改为 `grid.cell_center_point(i, j, k)`

2. 生成 wrench 作用点时：
   - 从 `grid.stacked_points()` 改为 `grid.stacked_cell_centers()`

3. consistent traction / projected point reconstruction 中：
   - 基点改为 `grid.cell_center_point(...)`
   - `projected_offset` 也以 cell center 为参考点计算

4. torque accumulation 中：
   - lever arm 也统一基于 `stacked_cell_centers()`

#### 修改目标
让 native-band 这一整条链路从“采样 → mask → 力 → 力矩”都用同一套 cell-centered 几何语义。

#### 迁移要点
迁移时不要只改入口采样点；要确保下面这些地方是一起迁移的：

- field sampling
- projected point reconstruction
- force density / torque density accumulation
- 所有 `points` / `reference` / `lever` 的配对逻辑

#### 这一步解决什么问题
这是实际消除 native-band 面积偏差和 wrench 偏差的主修改点。

---

### 4.3 `src/pfcsdf/dynamics/benchmarks.py`

#### 修改内容
在 native-band sphere / flat benchmark 中，构造 `extra_mask` 时把：

- `grid.stacked_points()`

改成：

- `grid.stacked_cell_centers()`

#### 修改目标
让 benchmark 级别的 footprint / radial-limit mask 与底层 native-band 累加器使用同一套 cell center 语义。

#### 迁移要点
如果本地分支还有其他 benchmark 也在外层构造 support mask，同样要检查是否仍在用 `stacked_points()`；只要这个 mask 代表“哪些体素参与累加”，它就应当改成 `stacked_cell_centers()`。

#### 这一步解决什么问题
即使底层累加器改对了，如果 benchmark 的 `extra_mask` 仍按角点判定，边界面积依然会算错。

---

### 4.4 `configs/main_tables.yaml`

#### 修改内容
把 native-band flat 的 grid shape 从：

- `[5, 5, 21]`

改成：

- `[4, 4, 20]`

`origin` 保持：

- `[-0.4, -0.4, -0.1]`

`spacing` 保持：

- `[0.2, 0.2, 0.01]`

#### 修改目标
在 **cell-centered** 语义下，让 `0.8 x 0.8` footprint 正好对应 `4 x 4` 个 cell centers。

#### 为什么要这样改
修复前的 `5 x 5 x 21` 实际上会把边界也吃进去，隐含支撑面积会膨胀到 `1.00`，而真实 footprint 面积是 `0.64`。

修复后：

- x/y 方向 `4 x 4` 个 cells × `0.2 x 0.2` 面积 = `0.64`
- z 方向 `20` 个 cells × `0.01` = `0.20`

这与目标 box bounds 完全对齐。

#### 迁移要点
这里最容易犯的错是：

- 把 `origin` 也一起改成 `[-0.3, -0.3, -0.095]`

**不要这么做。**

最终系统化修复后的约定是：

- `origin` 仍表示 **grid bounds lower corner**；
- `cell_center_origin = origin + 0.5 * spacing` 由代码推导出来。

---

### 4.5 `configs/efficiency.yaml`

#### 修改内容
和 `main_tables.yaml` 一样，把 native-band 对应 grid shape 从 `[5, 5, 21]` 改为 `[4, 4, 20]`。

#### 修改目标
保持 efficiency table 与主表使用一致的几何离散定义，避免“主表修好了，但效率表仍在旧网格上跑”。

---

### 4.6 `src/pfcsdf/experiments/ablation.py`

#### 修改内容
把 native-band flat 的默认 `UniformGrid3D` 从：

- `shape=(5, 5, 21)`

改成：

- `shape=(4, 4, 20)`

`origin` 和 `spacing` 维持为：

- `origin=(-0.4, -0.4, -0.1)`
- `spacing=(0.2, 0.2, 0.01)`

#### 修改目标
让论文主 ablation 的 native-band 结果和修复后的 cell-centered 语义保持一致。

#### 迁移要点
如果本地还有其他地方手写了 native-band flat 的 grid 定义，也要一起搜一遍，避免出现“表格 A 用 4x4x20，表格 B 还在用 5x5x21”的分叉。

---

## 5. 测试层做了哪些修改，以及目的是什么

### 5.1 新增 `tests/test_step_40_cell_centered_native_band.py`

#### 新增内容
新增了一组专门防回归的测试：

1. 检查 `cell_center_origin` 和 `cell_center_point(...)` 是否真的是半个 spacing 偏移；
2. 检查 native-band flat 的有效支撑面积是否回到了真实 footprint 面积；
3. 检查静态压入时得到的 force 是否和解析预期一致。

#### 修改目标
把这次修复的核心假设固化成回归测试，避免以后再有人把 `point()` 和 `cell_center_point()` 混回去。

---

### 5.2 修改 `tests/test_step_21_native_band_flat.py`

#### 修改内容
把原来写死为“origin = 第一个 cell center”的构造改回：

- `origin = grid bounds lower corner`

例如从：

- `-0.5 * n * dx + 0.5 * dx`

改成：

- `-0.5 * n * dx`

#### 修改目标
让测试与新的统一语义一致：`origin` 表示 bounds lower corner，而不是第一个 cell center。

---

### 5.3 修改 `tests/test_step_30_dynamics_native_band.py`
### 5.4 修改 `tests/test_step_33_native_band_continuity_update.py`
### 5.5 修改 `tests/test_step_36_impulse_corrected_midpoint.py`
### 5.6 修改 `tests/test_step_37_work_consistent_midpoint.py`
### 5.7 修改 `tests/test_step_38_consistent_traction_reconstruction.py`
### 5.8 修改 `tests/test_step_39_paper_ablation.py`

#### 共同修改内容
这些测试主要做了两类修正：

1. 把 native-band flat 的 grid shape 从 `(5, 5, 21)` 改成 `(4, 4, 20)`；
2. 把少数依赖旧几何假设的断言，从“强行要求某个旧数值完全不变”改成了更稳妥的“误差不恶化 / 不显著回退”。

#### 修改目标
这些测试原本有一部分是在保护“旧错误几何”下的历史输出。修复体素语义后，测试应该保护的是：

- 新语义的一致性；
- 关键误差指标不恶化；
- 修复后 force / impulse / state 指标至少不比修复前更差。

#### 迁移要点
迁移时如果本地 codex 发现某些测试在修复后数值发生变化，不要机械地把新代码改回旧值。应先判断：

- 这个断言是在保护“方法正确性”，还是在保护“旧错误输出”；
- 若是后者，应改成更有物理含义的判据。

---

## 6. 论文描述层的同步修改（可选，但建议做）

### 文件
- `pfc_sdf_dynamics_paper_sprint1.tex`
- `pfc_sdf_dynamics_paper_sprint3_appendix.tex`

### 修改内容
在正文方法描述和 reference comparison 说明中加入这句：

> *For all native-band experiments in this paper, native-band now uses cell-centered volumetric sampling and cell-centered footprint support masking.*

### 修改目标
让论文中的方法定义与修复后的代码实现完全一致，避免文稿仍沿用旧口径。

---

## 7. 迁移给本地 codex 时，建议明确传达的约束

建议把下面几条直接作为迁移要求告诉本地 codex：

1. **以 `fixed/` 为唯一真值来源。**
2. `UniformGrid3D.origin` 必须继续表示 **grid bounds lower corner**。  
   不要把整个项目改成“origin = first cell center”。
3. 新增 `cell_center_*` API，而不是破坏旧 `point()` 语义。  
   这样可以把修复影响控制在 native-band 相关路径上。
4. 所有 native-band 的：
   - SDF sampling
   - support mask
   - force accumulation
   - torque accumulation
   必须统一按 **cell center** 处理。
5. 只要 grid / mask 代表的是“哪些体素参与积分”，它就应该基于 **cell_center_point / stacked_cell_centers**。
6. 迁移结束后，必须重跑：
   - native-band flat 相关测试；
   - 主表生成脚本；
   - efficiency 表生成脚本；
   - 论文 PDF 编译。

---

## 8. 建议的迁移后验收标准

迁移完成后，至少确认下面几点：

### 代码层
- `UniformGrid3D` 提供了明确的 `cell_center_*` API；
- native-band 路径不再使用 `point()` / `stacked_points()` 作为体素积分点位；
- benchmark 侧的 `extra_mask` 已切到 `stacked_cell_centers()`。

### 配置层
- native-band flat 默认 grid 为：
  - `origin = [-0.4, -0.4, -0.1]`
  - `spacing = [0.2, 0.2, 0.01]`
  - `shape = [4, 4, 20]`

### 结果层
native-band 主表应当恢复到明显更合理的量级，例如：

- `+ work consistency` 或 `+ consistent traction reconstruction` 的 `peak force error` 显著下降；
- `release timing error` 显著小于旧版本；
- `native_band_force_time.pdf` 不再表现为明显过高、过尖、过早结束的曲线。

### 文稿层（如果同步迁移）
论文里已明确写出：

> *native-band now uses cell-centered volumetric sampling and cell-centered footprint support masking*

---

## 9. 最后一句给本地 codex 的摘要指令

如果你想给本地 codex 一个短版本任务说明，可以直接用下面这段：

> 请以 `fixed/` 目录为真值，将 native-band 的体素采样语义迁移为统一的 cell-centered volumetric sampling。保留 `UniformGrid3D.point()` 的旧 corner 语义，新增并使用 `cell_center_point()` / `stacked_cell_centers()`。把 native-band 的 SDF sampling、footprint support mask、force/torque accumulation 全部切换到 cell center。同步把 native-band flat 的默认 grid 从 `(5,5,21)` 改为 `(4,4,20)`，但保留 `origin = bounds lower corner` 的直观配置写法。最后更新相关测试、主表、效率表和论文方法描述，确保项目不再依赖手工调整 origin 来修复体素问题。

