# Pipeline 规范（00_pipeline）

> 本文件补充根目录 claude.md 中与 pipeline 相关的部分，其余规范继续遵守根目录 claude.md。

---

## Pipeline 核心思想

每个 pipeline 是一条流水线：**原始数据 → 步骤1 → 步骤2 → ... → 最终产物**。

每一步职责单一，输入输出清晰，可独立验证，可单独重跑。

---

## Notebook 结构（强制顺序）

每个 pipeline 的 `.ipynb` 必须按以下顺序组织 cell：

```
# ── 0. 配置区 ──────────────────────────────────────────────
  所有路径、参数、开关集中在这里，方便Master统一修改

# ── 1. 获取批量路径 ─────────────────────────────────────────
  通过代码扫描/筛选，打印出待处理的文件路径列表
  执行后由Master人工确认，手动复制路径到下方配置区再继续

# ── 2. Step N：[动词 + 具体操作] ────────────────────────────
  输入：来自哪里
  处理：做了什么
  输出：存到哪里
  （每个步骤一个独立 cell 块，步骤之间加分隔注释）

# ── 3. 验证 ────────────────────────────────────────────────
  用 display() 抽查输出样本，确认结果符合预期
```

---

## 配置区写法

```python
# ── 配置 ──────────────────────────────────────────────────

# 路径
raw_data_dir        = "../01_data/raw/"
pipeline_output_dir = "../01_data/processed/step1_output/"

# 参数
batch_size          = 32
target_image_size   = (640, 640)

# 开关
enable_debug_mode   = False
```

- 所有需要改动的变量只在这里出现一次，下方代码直接引用
- 禁止在步骤 cell 里硬编码路径或参数

---

## 批量路径获取规范

第一个执行 cell 负责输出候选路径，格式要让Master方便复制：

```python
from pathlib import Path

sub_dir_path = ""

all_file_path_list = sorted(Path(raw_data_dir).glob(f"*/{sub_dir_path}"))

# 打印供主人确认，格式可直接粘贴进 list
for file_path in all_file_path_list:
    print(f'    "{file_path}",')

print(f"\n共 {len(all_file_path_list)} 个文件夹")
```

Master确认后，将路径粘贴回配置区的 `target_file_path_list`，再运行后续步骤。

---

## 输入输出文件夹规范

每个步骤的输入输出都有对应文件夹，结构如下：

```
00_pipeline/
  └── pipeline_name/
        ├── claude.md              # 本 pipeline 的专属规范（本文件）
        ├── pipeline_name.ipynb    # 主流程 notebook
        ├── step1_input_example/   # step1 输入样例
        ├── step1_output_example/  # step1 输出样例
        ├── step2_input_example/
        ├── step2_output_example/
        └── ...
```

**样例要求：**
- 每个 example 文件夹放 1～3 个真实样本，能直观看出输入/输出长什么样
- 如果样本不方便保存（体积大、格式特殊、需要脱敏），改为放一个 `README.md` 简短描述：
  - 数据是什么格式
  - 典型样本长什么样（可以是文字描述或截图）
  - 处理后变成什么

---

## 步骤 Cell 写法模板

```python
# ── Step 1：[动词 + 具体操作，例如：裁剪图像到目标尺寸] ──────

# 输入：raw_data_dir 下的原始图像（.jpg）
# 输出：step1_output_dir 下裁剪后的图像

logger.info("Step 1 开始：裁剪图像")

os.makedirs(step1_output_dir, exist_ok=True)

for raw_image_path in target_file_path_list:
    try:
        cropped_image = crop_image_to_target_size(raw_image_path, target_image_size)
        save_image(cropped_image, step1_output_dir)
    except Exception as e:
        logger.error(f"裁剪失败 {raw_image_path}: {e}")

logger.info(f"Step 1 完成，共处理 {len(target_file_path_list)} 张")
```

每个cell需要开头有logger.info()结尾有logger.info()。里面写开始和结束信息。帮助在log里知道运行的哪个cell。干了什么。得到什么。

---

## 可读性优先原则（pipeline 特化）

- 步骤之间必须有 `# ── Step N：...` 分隔注释，一眼看出流程推进
- 每个步骤 cell 开头两行注释说明输入和输出来源
- 禁止跨步骤共用中间变量（每步输出写到磁盘，下一步从磁盘读）
- 每个步骤跑完打一条 `logger.info` 进度日志

---

*Master如有针对具体 pipeline 的额外规范，在各子文件夹的 claude.md 里追加即可*