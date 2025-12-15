import streamlit as st
import yaml
import tempfile
from pathlib import Path
import os
import time

# ==============================================================================
# 🛠️ 模拟后端函数导入 (实际使用时，请将你的 notebook 转为 .py 并在此导入)
# 例如: 
from step01_raw_image_seg_prediction import run_pipeline as run_step1
from step02_remove_duplicate_predictions import run_pipeline as run_step2
from step03_cut_out_the_object_in_the_image_and_then_perform_inference import run_pipeline as run_step3
# ==============================================================================
def mock_run_pipeline(config_path, step_name):
    """模拟运行过程，替换为你真实的 pipeline 调用"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    st.info(f"正在加载配置: {config_path}")
    st.write("🔧 运行参数预览:", cfg)
    
    with st.status(f"🚀 正在执行 {step_name}...", expanded=True) as status:
        st.write("加载模型中...")
        time.sleep(1)
        st.write("处理数据中 (模拟)...")
        time.sleep(2)
        status.update(label=f"✅ {step_name} 完成!", state="complete", expanded=False)
    
    return True

# ==============================================================================
# 🎨 辅助函数：加载和保存 YAML
# ==============================================================================
def load_yaml(path):
    if not Path(path).exists():
        st.error(f"❌ 找不到配置文件: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_temp_yaml(config_dict):
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
    yaml.dump(config_dict, tmp)
    return Path(tmp.name)

# ==============================================================================
# 🖥️ 主界面逻辑
# ==============================================================================
st.set_page_config(page_title="YOLO Pipeline Wizard", layout="wide", page_icon="🕵️")

# --- Sidebar: 全局状态与导航 ---
st.sidebar.title("🕵️ SWD Analysis")

# 1. 全局数据根目录 (Shared State)
if 'root_dir' not in st.session_state:
    # 读取 01_config.yaml 的默认值作为初始值
    base_cfg = load_yaml("config/01_config.yaml")
    st.session_state.root_dir = base_cfg['pipeline']['input']['root_directory'] if base_cfg else ""

st.sidebar.header("📂 全局设置")
# 当用户在这里修改，所有步骤的 root_dir 都会自动更新
st.session_state.root_dir = st.sidebar.text_input(
    "数据根目录 (Root Directory)", 
    value=st.session_state.root_dir,
    help="所有步骤将默认在该目录下寻找数据"
)

st.sidebar.markdown("---")
step = st.sidebar.radio(
    "流程导航", 
    ["1️⃣ 图像分割 (Segmentation)", 
     "2️⃣ 结果去重 (Deduplication)", 
     "3️⃣ 姿态与斑点 (Pose & Dot)", 
     "👁️ 结果可视化 (Inspector)"]
)

# ==============================================================================
# 🟢 Step 1: Segmentation (对应 01_config.yaml)
# ==============================================================================
if "1️⃣" in step:
    st.title("🧩 Step 1: Tiled Segmentation")
    st.markdown("基于 YOLO 分割模型对大图进行切片推理。")
    
    default_cfg = load_yaml("config/01_config.yaml")
    if default_cfg:
        with st.form("step1_form"):
            # 分区 1: 模型设置
            st.subheader("🤖 模型参数")
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                model_path = st.text_input("模型路径", default_cfg['yolo']['model_path'])
            with c2:
                device = st.text_input("Device", str(default_cfg['yolo']['device']))
            with c3:
                batch = st.number_input("Batch", value=default_cfg['yolo']['batch_size'])
            
            c4, c5 = st.columns(2)
            with c4:
                conf = st.slider("Confidence", 0.0, 1.0, default_cfg['yolo']['confidence_threshold'])
            with c5:
                iou = st.slider("IoU Threshold", 0.0, 1.0, default_cfg['yolo']['iou_threshold'])

            # 分区 2: 切片设置 (折叠起来，因为平时不常改)
            with st.expander("🖼️ 切片策略 (Processing Config)"):
                c6, c7 = st.columns(2)
                tile_h = c6.number_input("Tile Height", value=default_cfg['processing']['tile_height'])
                overlap = c7.slider("Overlap Ratio", 0.0, 0.9, default_cfg['processing']['overlap_ratio'])

            # 分区 3: 路径 (自动填充全局 root)
            st.subheader("📁 输入/输出")
            current_root = st.text_input("当前处理目录", value=st.session_state.root_dir, disabled=True)
            search_depth = st.number_input("搜索深度", value=default_cfg['pipeline']['input']['search_depth'])

            submit = st.form_submit_button("🚀 运行 Step 1", type="primary")

        if submit:
            # 组装参数
            run_cfg = default_cfg.copy()
            run_cfg['yolo'].update({'model_path': model_path, 'device': device, 'confidence_threshold': conf, 'iou_threshold': iou})
            run_cfg['processing'].update({'tile_height': tile_h, 'tile_width': tile_h, 'overlap_ratio': overlap})
            run_cfg['pipeline']['input']['root_directory'] = st.session_state.root_dir # 使用全局变量
            run_cfg['pipeline']['input']['search_depth'] = search_depth
            run_cfg['pipeline']['logging']['force'] = True

            tmp_path = save_temp_yaml(run_cfg)
            
            # --- 调用真实函数 ---
            # run_step1(config_path=tmp_path) 
            # mock_run_pipeline(tmp_path, "Segmentation")
            run_step1(config_path=tmp_path)
            
            st.success("Step 1 完成！请前往左侧导航进入 Step 2。")

# ==============================================================================
# 🔵 Step 2: Deduplication (对应 02_config.yaml)
# ==============================================================================
elif "2️⃣" in step:
    st.title("🧹 Step 2: Result Deduplication")
    st.markdown("对切片产生的重复检测框进行 NMS 去重。")

    default_cfg = load_yaml("config/02_config.yaml")
    if default_cfg:
        with st.form("step2_form"):
            st.subheader("⚙️ 去重策略")
            c1, c2 = st.columns(2)
            method = c1.selectbox("去重方法", ["NMS", "NMM", "GREEDYNMM"], index=0)
            metric = c2.selectbox("重叠度量", ["IOU", "IOS"], index=1) # 默认 IOS
            
            thresh = st.slider("重叠阈值 (Overlap Threshold)", 0.0, 1.0, default_cfg['processing']['overlap_threshold'])
            
            st.subheader("📁 待处理目录")
            st.info(f"将处理以下根目录下的所有数据: **{st.session_state.root_dir}**")
            # 你的 02_config.yaml 里是一个 list，这里我们可以简化为只处理当前的 root_dir
            # 或者提供一个 Text Area 让用户输入多个路径
            
            submit = st.form_submit_button("🚀 运行 Step 2", type="primary")
        
        if submit:
            run_cfg = default_cfg.copy()
            run_cfg['processing'].update({'method': method, 'overlap_metric': metric, 'overlap_threshold': thresh})
            # 强制覆盖列表为当前的单一目录，或者你可以保留原逻辑
            run_cfg['pipeline']['root_dir_path_list'] = [st.session_state.root_dir]
            
            tmp_path = save_temp_yaml(run_cfg)
            
            # --- 调用真实函数 ---
            # run_step2(config_path=tmp_path)
            mock_run_pipeline(tmp_path, "Deduplication")
            st.success("Step 2 完成！")

# ==============================================================================
# 🟠 Step 3: Pose & Dot (对应 03_config.yaml)
# ==============================================================================
elif "3️⃣" in step:
    st.title("💃 Step 3: Pose & Dot Inference")
    st.markdown("从原图中抠出目标 (Crop)，分别进行姿态估计和斑点检测。")

    default_cfg = load_yaml("config/03_config.yaml")
    if default_cfg:
        with st.form("step3_form"):
            st.info(f"数据源: {st.session_state.root_dir}")
            
            st.subheader("🧠 模型配置")
            
            col_pose, col_dot = st.columns(2)
            
            # 左列：Pose 模型
            with col_pose:
                st.markdown("#### 💃 Pose Model")
                pose_model = st.text_input("Pose Path", default_cfg['models']['pose_model'])
                pose_conf = st.slider("Pose Conf", 0.0, 1.0, default_cfg['pose_args']['conf'])
            
            # 右列：Dot 模型
            with col_dot:
                st.markdown("#### 🐞 Dot Model")
                dot_model = st.text_input("Dot Path", default_cfg['models']['dot_model'])
                dot_conf = st.slider("Dot Conf", 0.0, 1.0, default_cfg['dot_args']['conf'])

            submit = st.form_submit_button("🚀 运行 Step 3", type="primary")
            
        if submit:
            run_cfg = default_cfg.copy()
            run_cfg['data_root'] = st.session_state.root_dir
            run_cfg['models']['pose_model'] = pose_model
            run_cfg['models']['dot_model'] = dot_model
            run_cfg['pose_args']['conf'] = pose_conf
            run_cfg['dot_args']['conf'] = dot_conf
            
            tmp_path = save_temp_yaml(run_cfg)
            
            # --- 调用真实函数 ---
            # run_step3(config_path=tmp_path)
            mock_run_pipeline(tmp_path, "Pose & Dot")
            st.success("Pipeline 全部完成！")

# ==============================================================================
# 👁️ Inspector: 简单的结果可视化
# ==============================================================================
elif "👁️" in step:
    st.title("👁️ 结果审查 (Inspector)")
    
    # 简单的文件浏览逻辑
    target_dir = st.session_state.root_dir
    if os.path.exists(target_dir):
        # 假设你想看 output 目录下的图
        # 这里需要你根据实际生成逻辑写一点点代码来寻找生成的图片或带框图
        st.warning("可视化功能需要连接到具体的输出目录结构。")
        st.write(f"当前关注目录: {target_dir}")
        
        # 示例：列出 raw_data 下的图
        # images = list(Path(target_dir).rglob("*.jpg"))
        # if images:
        #     selected_img = st.selectbox("选择图片", images)
        #     st.image(str(selected_img), caption="原始图片")
    else:
        st.error("目录不存在，请先在 Step 1 配置正确的路径。")