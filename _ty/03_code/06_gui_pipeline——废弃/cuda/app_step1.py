import streamlit as st
from pathlib import Path
import yaml
import traceback

# ✅ 替换成你真实的文件名
from step01_raw_image_seg_prediction import run_pipeline as run_step1

# ---------------------------------------------------------------------
# 基本配置
# ---------------------------------------------------------------------
CONFIG_PATH = Path("config/01_config.yaml")

st.set_page_config(
    page_title="Step 1 - Segmentation Config & Runner",
    layout="wide",
    page_icon="🧩",
)

st.title("🧩 Step 1：Segmentation 配置 & 运行面板")
st.caption("编辑 YAML 配置 → 保存 → 一键运行原来的 YOLO Segmentation pipeline。")


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def load_yaml_text(path: Path) -> str:
    if not path.exists():
        return "# 配置文件不存在，请先创建：\n" + str(path)
    return path.read_text(encoding="utf-8")


def normalize_yaml(yaml_text: str) -> tuple[dict, str]:
    """
    把 YAML 文本 parse 一下，如果合法：
    - 返回 (dict, 重新 dump 后的漂亮 YAML)
    - 保留中文 & key 顺序
    """
    data = yaml.safe_load(yaml_text)
    pretty = yaml.dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return data, pretty


# ---------------------------------------------------------------------
# Session 初始化
# ---------------------------------------------------------------------
if "yaml_text" not in st.session_state:
    st.session_state.yaml_text = load_yaml_text(CONFIG_PATH)

if "last_saved" not in st.session_state:
    st.session_state.last_saved = None

# ---------------------------------------------------------------------
# 左右布局：左边编辑 YAML，右边保存 & 运行
# ---------------------------------------------------------------------
left, right = st.columns([2.5, 1.5])

with left:
    st.subheader("📝 配置文件编辑")

    st.markdown(f"- 配置路径：`{CONFIG_PATH}`")

    # 优先用 code_editor（高亮 + 行号），没有就退回 text_area
    if hasattr(st, "code_editor"):
        new_text = st.code_editor(
            st.session_state.yaml_text,
            language="yaml",
            height=500,
            key="yaml_editor",
        )
    else:
        new_text = st.text_area(
            "YAML 内容",
            value=st.session_state.yaml_text,
            height=500,
            key="yaml_editor",
        )

    # 保持最新编辑值在 session_state
    st.session_state.yaml_text = new_text

with right:
    st.subheader("💾 保存 / 🚀 运行")

    # --- 保存按钮 ---
    if st.button("💾 保存配置到磁盘", use_container_width=True):
        try:
            cfg_dict, pretty = normalize_yaml(st.session_state.yaml_text)
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(pretty, encoding="utf-8")
            st.session_state.yaml_text = pretty
            st.session_state.last_saved = str(CONFIG_PATH)
            st.success(f"配置已保存到：{CONFIG_PATH}")
        except Exception as e:
            st.error("❌ YAML 解析或保存失败，请检查缩进/冒号等语法。")
            st.exception(e)

    if st.session_state.last_saved:
        st.caption(f"最近保存：`{st.session_state.last_saved}`")

    st.markdown("---")

    # --- 运行按钮 ---
    run_clicked = st.button("🚀 使用当前配置运行 pipeline", type="primary", use_container_width=True)

    if run_clicked:
        # 先尝试解析一次，防止 YAML 有问题
        try:
            cfg_dict, pretty = normalize_yaml(st.session_state.yaml_text)
        except Exception as e:
            st.error("❌ 当前 YAML 无法解析，请先修好再运行。")
            st.exception(e)
        else:
            # 解析 OK 时，先保存一份再运行
            CONFIG_PATH.write_text(pretty, encoding="utf-8")
            st.session_state.yaml_text = pretty
            st.session_state.last_saved = str(CONFIG_PATH)

            # 从配置里抽出一些关键信息，用于后面提示输出路径
            try:
                root_dir = cfg_dict["pipeline"]["input"]["root_directory"]
                search_depth = cfg_dict["pipeline"]["input"]["search_depth"]
                out_dir_name = cfg_dict["pipeline"]["output"]["directory_name"]
                out_file_name = cfg_dict["pipeline"]["output"]["file_name"]
            except Exception:
                root_dir = None
                search_depth = None
                out_dir_name = None
                out_file_name = None

            with st.status("🚀 正在执行 YOLO Segmentation pipeline...", expanded=True) as status:
                status.write(f"使用配置文件：`{CONFIG_PATH}`")

                try:
                    # 真正调用你原来的代码
                    run_step1(config_path=CONFIG_PATH)

                    status.update(
                        label="✅ 执行完成",
                        state="complete",
                        expanded=False,
                    )

                    st.success("Pipeline 已执行完成 ✅")

                    # 输出路径提示（模式级别）
                    if root_dir and out_dir_name and out_file_name:
                        st.markdown("#### 📁 输出结果位置（模式）")
                        st.code(
                            f"{root_dir}/**/{out_dir_name}/{out_file_name}",
                            language="bash",
                        )
                        st.caption("可以在终端里使用 `find` 或 `ls` 查看实际生成的 JSON。")
                    else:
                        st.info("已完成执行，但无法从 YAML 中解析输出路径字段。")

                except Exception as e:
                    status.update(
                        label="❌ 执行失败",
                        state="error",
                        expanded=True,
                    )
                    st.error("运行过程中出现异常：")
                    st.exception(e)

    # 小提示
    st.markdown("---")
    st.markdown(
        "💡 建议：日志文件仍然按你原来的 `pipeline.logging.log_file_name` 配置存放，"
        "需要查看详细进度/错误时可以直接打开对应 log。"
    )
