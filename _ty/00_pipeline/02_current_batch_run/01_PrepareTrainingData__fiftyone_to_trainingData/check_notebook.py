"""
检查 Notebook 文件是否符合规范
"""
import json
from pathlib import Path
import sys

def check_notebook(file_path):
    """检查单个 Notebook 文件"""
    print(f"\n{'='*80}")
    print(f"检查文件: {file_path.name}")
    print(f"{'='*80}")

    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    issues = []

    # 1. 检查是否包含日志配置
    has_logging_config = False
    has_ipynbname_import = False
    has_logger = False

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'import logging' in source:
                has_logging_config = True
            if 'import ipynbname' in source:
                has_ipynbname_import = True
            if 'logger = logging.getLogger' in source:
                has_logger = True

    if not has_logging_config:
        issues.append("❌ 缺少日志配置：没有导入 logging 模块")
    if not has_ipynbname_import:
        issues.append("❌ 缺少 ipynbname 导入：无法获取当前 notebook 文件名")
    if not has_logger:
        issues.append("❌ 缺少 logger 配置：没有创建 logger 对象")

    # 2. 检查是否使用 print() 展示结构化内容
    print_usage_issues = []
    for i, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            # 检查是否有 print() 用于展示数据
            if 'print(' in source and ('df' in source or 'data' in source or 'result' in source):
                print_usage_issues.append(f"Cell {i+1}: 可能使用 print() 展示结构化内容")

    if print_usage_issues:
        issues.append("⚠️  可能使用 print() 展示结构化内容（应使用 IPython.display）")
        for issue in print_usage_issues[:3]:  # 只显示前3个
            issues.append(f"   {issue}")

    # 3. 检查 cell 职责是否单一
    cell_count = len(notebook.get('cells', []))
    code_cells = [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']

    # 检查是否有超过30行的函数
    large_function_issues = []
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        lines = source.split('\n')
        # 简单检查函数定义
        in_function = False
        function_lines = 0
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                in_function = True
                function_lines = 1
            elif in_function and (line.strip().startswith('def ') or line.strip().startswith('async def ')):
                # 新函数开始
                if function_lines > 30:
                    large_function_issues.append(f"Cell {i+1}: 函数超过30行（{function_lines}行）")
                function_lines = 1
            elif in_function and line.strip() and not line.strip().startswith('#'):
                function_lines += 1
            elif in_function and line.strip() == '':
                function_lines += 1
            elif in_function and line.strip().startswith('#'):
                function_lines += 1
            elif in_function and not line.strip():
                function_lines += 1
            elif in_function and line.strip().startswith('    ') or line.strip().startswith('\t'):
                function_lines += 1
            else:
                in_function = False
                if function_lines > 30:
                    large_function_issues.append(f"Cell {i+1}: 函数超过30行（{function_lines}行）")
                function_lines = 0

        # 检查最后一个函数
        if in_function and function_lines > 30:
            large_function_issues.append(f"Cell {i+1}: 函数超过30行（{function_lines}行）")

    if large_function_issues:
        issues.append("❌ 有函数超过30行（应抽到独立 .py 文件）")
        for issue in large_function_issues[:3]:
            issues.append(f"   {issue}")

    # 4. 检查是否有 .log 文件
    log_file = file_path.with_suffix('.log')
    if not log_file.exists():
        issues.append("❌ 缺少配套的 .log 文件")

    # 5. 检查是否符合 pipeline 结构规范
    # 检查是否有配置区、批量路径获取、步骤、验证等结构
    has_config_section = False
    has_batch_path_section = False
    has_step_sections = False
    has_verification_section = False

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if '配置' in source or '配置区' in source:
                has_config_section = True
            if '批量路径' in source or '获取批量路径' in source:
                has_batch_path_section = True
            if 'Step' in source or '步骤' in source:
                has_step_sections = True
            if '验证' in source or '检查' in source:
                has_verification_section = True

    if not has_config_section:
        issues.append("⚠️  缺少明确的配置区标记")
    if not has_batch_path_section:
        issues.append("⚠️  缺少批量路径获取区标记")
    if not has_step_sections:
        issues.append("⚠️  缺少步骤区标记")
    if not has_verification_section:
        issues.append("⚠️  缺少验证区标记")

    # 6. 检查错误处理
    error_handling_issues = []
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        # 检查是否有裸 except Exception
        if 'except Exception' in source and 'except Exception as e:' not in source:
            error_handling_issues.append(f"Cell {i+1}: 可能使用裸 except Exception")

    if error_handling_issues:
        issues.append("⚠️  可能存在裸 except Exception（应指定具体异常类型）")
        for issue in error_handling_issues[:3]:
            issues.append(f"   {issue}")

    # 7. 检查命名规范
    naming_issues = []
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        # 检查模糊命名
        bad_names = ['data', 'info', 'tmp', 'res', 'result']
        for bad_name in bad_names:
            # 简单检查变量名
            if f' {bad_name} ' in source or f' {bad_name}=' in source or f'({bad_name},' in source:
                naming_issues.append(f"Cell {i+1}: 可能使用模糊命名 '{bad_name}'")

    if naming_issues:
        issues.append("⚠️  可能存在模糊命名（应使用详细变量名）")
        for issue in naming_issues[:3]:
            issues.append(f"   {issue}")

    # 8. 检查是否使用 from pathlib import Path
    has_pathlib_import = False
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'from pathlib import Path' in source:
                has_pathlib_import = True
                break

    if not has_pathlib_import:
        issues.append("⚠️  未使用 from pathlib import Path（推荐使用）")

    # 输出结果
    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ Notebook 文件符合规范")

    return len(issues) == 0

def main():
    """主函数"""
    pipeline_dir = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData")

    # 查找所有 .ipynb 文件
    notebook_files = list(pipeline_dir.glob("*.ipynb"))

    if not notebook_files:
        print("❌ 未找到任何 .ipynb 文件")
        return

    print(f"找到 {len(notebook_files)} 个 Notebook 文件:")
    for nb_file in notebook_files:
        print(f"  - {nb_file.name}")

    all_passed = True
    for nb_file in notebook_files:
        passed = check_notebook(nb_file)
        all_passed = all_passed and passed

    print(f"\n{'='*80}")
    if all_passed:
        print("✅ 所有 Notebook 文件都符合规范")
    else:
        print("❌ 部分 Notebook 文件不符合规范")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()