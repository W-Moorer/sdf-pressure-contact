# sdf-pressure-contact

基于 Signed Distance Function (SDF) 的压力场接触验证仓库。当前仓库的主体是 `pfc_sdf_validation/` 子项目，它不是一个通用工程模板，而是一个偏研究/验证性质的 Python 代码库，用来实现、对比和复现实验性的 PFC-SDF 接触与动力学方案。

## 项目定位

这个仓库目前主要包含四类内容：

- `src/pfcsdf/` 中的核心算法代码
- `tests/` 中按步骤累积的回归测试
- `experiments/`、`configs/` 中的实验入口与配置
- `results/`、论文 `tex/pdf/csv` 等研究产物

从代码实现看，项目重点不在传统网格接触重建，而在一条更偏 SDF-native 的求解链路：

```text
SDF
-> depth field
-> pressure field
-> balance field h = p_A - p_B
-> native-band / sparse active-set traversal
-> local-normal or consistent-traction reconstruction
-> wrench accumulation
-> event-aware dynamics integration
```

## 当前代码覆盖范围

`pfc_sdf_validation/` 已经形成一个比较完整的验证闭环，主要覆盖：

- 静力接触基线
  - 平面均匀压入
  - 球-平面接触的解析解与数值积分对比
- SDF-native 接触积分
  - 体素网格上的深度/压力/平衡场采样
  - narrow-band / native-band 稀疏活动单元遍历
  - continuity-aware warm start 与边界更新
  - local normal correction
  - consistent traction reconstruction
  - wrench/torque 累积
- 动力学时间推进
  - midpoint / substep / event-aware midpoint
  - impulse-corrected 与 work-consistent 变体
- 复杂几何验证
  - 2D 复合支撑轮廓刚体下落
  - 3D support cloud 的 6-DoF 刚体接触基础能力
- 论文输出
  - CSV / Markdown / LaTeX 表格
  - PDF 图

## 仓库结构

```text
.
├─ README.md
├─ LICENSE
└─ pfc_sdf_validation/
   ├─ pyproject.toml
   ├─ README.md
   ├─ src/pfcsdf/
   │  ├─ contact/      # 接触区域、active set、native band、wrench 重建
   │  ├─ dynamics/     # 时间积分、事件控制器、复杂几何与 6-DoF 基准
   │  ├─ geometry/     # SDF primitive、体素网格、复杂体采样
   │  ├─ physics/      # depth/pressure 物理关系
   │  ├─ reporting/    # 表格与图导出
   │  └─ solvers/      # 静力与基础动力学求解器
   ├─ experiments/     # 论文与基准实验入口
   ├─ configs/         # YAML 配置
   ├─ tests/           # step_01 ~ step_57 的渐进式测试
   └─ results/         # 已生成的表格与图
```

## 快速开始

根目录本身不是 Python 包，安装和运行都在 `pfc_sdf_validation/` 下进行：

```bash
cd pfc_sdf_validation
python -m pip install -e .[dev]
```

当前 `pyproject.toml` 声明的主要依赖包括：

- Python `>=3.10`
- `numpy`
- `scikit-image`
- `pandas`
- `matplotlib`
- `PyYAML`
- `pytest`（开发依赖）

## 运行测试

```bash
cd pfc_sdf_validation
pytest
```

测试不是零散堆叠，而是按 `test_step_01_...` 到 `test_step_57_...` 的方式逐步扩展。目前仓库内共有 51 个测试文件，比较适合用来理解功能演进顺序和回归边界。

## 常用实验入口

在 `pfc_sdf_validation/` 目录下：

```bash
python experiments/run_main_tables.py
python experiments/run_efficiency_tables.py
python experiments/run_plot_suite.py
python experiments/run_complex_case.py
python experiments/run_complex_case_6dof.py
```

这些脚本会在 `results/tables/` 与 `results/figures/` 下生成或更新论文相关产物，例如：

- 主表与消融表
- continuity / traction 效率表
- flat / sphere / native-band 力学曲线图
- complex case 与 6-DoF 基准输出

## 阅读建议

如果你是第一次进入这个仓库，建议按下面顺序阅读：

1. `pfc_sdf_validation/pyproject.toml`
2. `pfc_sdf_validation/src/pfcsdf/__init__.py`
3. `pfc_sdf_validation/src/pfcsdf/contact/native_band.py`
4. `pfc_sdf_validation/src/pfcsdf/dynamics/benchmarks.py`
5. `pfc_sdf_validation/src/pfcsdf/experiments/paper_suite.py`
6. `pfc_sdf_validation/tests/`

这样可以先建立“接口层 -> 核心算法 -> 实验输出 -> 回归验证”的整体认知。

## 当前分析结论

基于现有代码，这个仓库更准确的描述应当是：

- 一个围绕 PFC-SDF 接触模型的验证型研究仓库，而不是面向生产的通用接触库
- 重点方法是 native-band 稀疏体积分与 continuity-aware 动力学推进，而不是显式表面网格重建
- 已经具备从静力、动力学、复杂几何到论文产物导出的完整实验闭环
- 6-DoF 部分已经有明确基础实现与输出脚本，但整体成熟度仍低于平面/轴对称/2D 复杂几何部分

如果需要更细的阶段说明、实验解释或论文导出细节，可以继续查看 `pfc_sdf_validation/README.md` 与对应的 `experiments/`、`tests/` 文件。
