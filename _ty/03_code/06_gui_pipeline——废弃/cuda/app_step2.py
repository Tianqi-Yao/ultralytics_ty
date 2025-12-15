import streamlit as st
import yaml
import tempfile
from pathlib import Path

# ==== 引入 Step 2 后端 ====
from step02_remove_duplicate_predictions import run_pipeline as run_step2


# ==============================================================================
# 🔧 YAML 工具函数
# ==============================================================================
def load_yaml(path: str | Path):
    path = Path(path)
    if not path.exists():
        st.error(f"❌ 找不到配置文件: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_temp_yaml(config_dict: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(config_dict, tmp, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def tail_file(path: Path, max_lines: int = 80) -> str:
    """读取日志文件最后 max_lines 行（可选功能）"""
    if not path.exists():
        return f"(日志文件不存在: {path})"
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception as e:
        return f"(读取日志失败: {e})"


def build_log_path(root_dir: str, cfg: dict) -> Path:
    """
    根据你的 logging 逻辑推断日志路径：
    - output_json_subpath: 'output/02_combined_annotations_dedup.json'
    - log_file_name: '02_dedup.log'
    日志放在 output_json 的父目录，也就是 root_dir / output / 02_dedup.log
    """
    root = Path(root_dir)
    out_subpath = Path(cfg["pipeline"]["output_json_subpath"])  # e.g. output/xxx.json
    out_dir = out_subpath.parent                               # e.g. output
    log_name = cfg["logging"]["log_file_name"]                 # e.g. 02_dedup.log
    return root / out_dir / log_name


# ==============================================================================
# 🖥️ Streamlit UI
# ==============================================================================
st.set_page_config(
    page_title="YOLO SWD – Step 2 Dedup",
    layout="wide",
    page_icon="🧹",
)

st.sidebar.title("🕵️ SWD Pipeline – Step 2")

# ---- 读取默认配置 ----
default_cfg = load_yaml("config/02_config.yaml")
if not default_cfg:
    st.stop()

# ---- 全局 root_dir：从 config 里拿第一个 root_dir_path_list 作为默认 ----
if "root_dir" not in st.session_state:
    root_list = default_cfg["pipeline"].get("root_dir_path_list") or []
    if root_list:
        st.session_state.root_dir = str(root_list[0])
    else:
        st.session_state.root_dir = ""

st.sidebar.header("📂 全局设置")
st.session_state.root_dir = st.sidebar.text_input(
    "数据根目录 (root_dir)",
    value=st.session_state.root_dir,
    help="会写入 pipeline.root_dir_path_list[0]，用于拼接输入/输出 JSON 路径",
)

st.title("🧹 Step 2: 去除重复预测 (Deduplication)")
st.caption("对 Step 1 生成的 COCO JSON 做去重（NMS / NMM / IOS 等）。")

# ---- 表单：让你改 processing 里的几个关键参数 ----
with st.form("step2_form"):
    st.subheader("⚙️ 去重策略")

    c1, c2 = st.columns(2)
    with c1:
        method = st.selectbox(
            "method",
            ["NMS", "NMM", "GREEDYNMM", "LSNMS"],
            index=["NMS", "NMM", "GREEDYNMM", "LSNMS"].index(
                default_cfg["processing"].get("method", "NMS").upper()
            ),
            help="去重方法：标准 NMS / 合并框 NMM 等。",
        )
    with c2:
        metric = st.selectbox(
            "overlap_metric",
            ["IOU", "IOS"],
            index=["IOU", "IOS"].index(
                default_cfg["processing"].get("overlap_metric", "IOS").upper()
            ),
            help="IOU：交并比；IOS：交 / 小框面积。",
        )

    c3, c4 = st.columns(2)
    with c3:
        thresh = st.slider(
            "overlap_threshold",
            0.0,
            1.0,
            float(default_cfg["processing"].get("overlap_threshold", 0.5)),
            help="当重叠度 ≥ 阈值时认为是重复框。",
        )
    with c4:
        class_agnostic = st.checkbox(
            "class_agnostic（跨类别去重）",
            value=bool(default_cfg["processing"].get("class_agnostic", False)),
            help="勾选后忽略 category_id，一律按同一类去重。",
        )

    st.subheader("📁 输入 / 输出 JSON")

    input_sub = default_cfg["pipeline"].get("input_json_subpath", "")
    output_sub = default_cfg["pipeline"].get("output_json_subpath", "")
    st.text_input(
        "input_json_subpath（相对 root_dir）",
        value=input_sub,
        disabled=True,
    )
    st.text_input(
        "output_json_subpath（相对 root_dir）",
        value=output_sub,
        disabled=True,
    )

    st.markdown(
        f"- 实际输入 JSON 将是：`{st.session_state.root_dir}/{input_sub}`  \n"
        f"- 实际输出 JSON 将是：`{st.session_state.root_dir}/{output_sub}`"
    )

    submit = st.form_submit_button("🚀 运行 Step 2", type="primary")

# ---- 点击运行 ----
if submit:
    if not st.session_state.root_dir:
        st.error("请先在左侧填写 root_dir。")
        st.stop()

    # 组装新的 config dict（浅拷贝足够）
    run_cfg = default_cfg.copy()

    # 更新 processing 部分
    run_cfg["processing"].update(
        dict(
            method=method,
            overlap_metric=metric,
            overlap_threshold=float(thresh),
            class_agnostic=bool(class_agnostic),
        )
    )

    # 更新 pipeline.root_dir_path_list 只用一个当前 root_dir
    run_cfg["pipeline"]["root_dir_path_list"] = [st.session_state.root_dir]

    # 其他字段（input_json_subpath / output_json_subpath / logging）保持原样

    # 保存到临时 YAML
    tmp_cfg_path = save_temp_yaml(run_cfg)

    # 推断日志路径（只针对当前 root_dir 的情况）
    log_path = build_log_path(st.session_state.root_dir, run_cfg)

    with st.status("🚀 正在执行 Step 2 (Deduplication)...", expanded=True) as status:
        status.write(f"使用配置文件：`{tmp_cfg_path}`")

        # 在这里就给一个 less +F 提示（方便你直接复制到终端里看实时日志）
        status.write("在终端中查看实时日志：")
        status.code(f"less +F {log_path}", language="bash")

        try:
            run_step2(config_path=tmp_cfg_path)
            status.update(
                label="✅ Step 2 完成！",
                state="complete",
                expanded=False,
            )
        except Exception as e:
            status.write("❌ 运行过程中出现异常：")
            st.exception(e)
            status.update(
                label="❌ Step 2 失败",
                state="error",
                expanded=True,
            )

    # 可选：在网页上简单预览日志最后几行
    st.subheader("📜 日志最后若干行（预览）")
    st.caption(f"日志文件：`{log_path}`")
    st.code(tail_file(log_path), language="bash")
