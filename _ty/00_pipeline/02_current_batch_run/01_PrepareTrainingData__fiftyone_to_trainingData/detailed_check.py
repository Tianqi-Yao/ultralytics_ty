"""
详细检查 Notebook 文件
"""
import json
from pathlib import Path

def analyze_notebook(file_path):
    """分析 Notebook 文件的详细结构"""
    print(f"\n{'='*80}")
    print(f"详细分析: {file_path.name}")
    print(f"{'='*80}")

    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    print(f"\n总共有 {len(cells)} 个 cells")

    # 统计 cell 类型
    markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    print(f"  - Markdown cells: {len(markdown_cells)}")
    print(f"  - Code cells: {len(code_cells)}")

    # 分析代码 cell 的行数
    print(f"\n代码 cell 行数统计:")
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        lines = source.split('\n')
        print(f"  Cell {i+1}: {len(lines)} 行")

    # 检查是否有日志相关代码
    print(f"\n日志相关检查:")
    has_logging_import = False
    has_ipynbname_import = False
    has_logger_creation = False
    has_log_file_handler = False

    for cell in code_cells:
        source = ''.join(cell.get('source', []))
        if 'import logging' in source:
            has_logging_import = True
        if 'import ipynbname' in source:
            has_ipynbname_import = True
        if 'logger = logging.getLogger' in source:
            has_logger_creation = True
        if 'logging.FileHandler' in source:
            has_log_file_handler = True

    print(f"  - 导入 logging: {'✅' if has_logging_import else '❌'}")
    print(f"  - 导入 ipynbname: {'✅' if has_ipynbname_import else '❌'}")
    print(f"  - 创建 logger: {'✅' if has_logger_creation else '❌'}")
    print(f"  - 配置 FileHandler: {'✅' if has_log_file_handler else '❌'}")

    # 检查是否有 .log 文件
    log_file = file_path.with_suffix('.log')
    print(f"\n日志文件检查:")
    print(f"  - 期望的 .log 文件: {log_file.name}")
    print(f"  - 是否存在: {'✅' if log_file.exists() else '❌'}")

    # 检查是否使用 print() 展示结构化内容
    print(f"\nprint() 使用检查:")
    print_issues = []
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        lines = source.split('\n')
        for j, line in enumerate(lines):
            if 'print(' in line and not line.strip().startswith('#'):
                # 检查是否是简单的打印
                if any(keyword in line for keyword in ['df', 'data', 'result', 'output', 'list']):
                    print_issues.append(f"Cell {i+1}, Line {j+1}: {line.strip()[:80]}")

    if print_issues:
        print(f"  发现 {len(print_issues)} 个可能的 print() 问题:")
        for issue in print_issues[:5]:
            print(f"    {issue}")
    else:
        print(f"  ✅ 未发现明显的 print() 问题")

    # 检查是否使用 IPython.display
    print(f"\nIPython.display 使用检查:")
    has_display = False
    for cell in code_cells:
        source = ''.join(cell.get('source', []))
        if 'from IPython.display' in source or 'IPython.display' in source:
            has_display = True
            break

    print(f"  - 使用 IPython.display: {'✅' if has_display else '❌'}")

    # 检查是否有配置区
    print(f"\n结构检查:")
    has_config = False
    has_batch_path = False
    has_steps = False
    has_verification = False

    for cell in cells:
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if '配置' in source or '配置区' in source:
                has_config = True
            if '批量路径' in source or '获取批量路径' in source:
                has_batch_path = True
            if 'Step' in source or '步骤' in source:
                has_steps = True
            if '验证' in source or '检查' in source:
                has_verification = True

    print(f"  - 配置区: {'✅' if has_config else '❌'}")
    print(f"  - 批量路径获取: {'✅' if has_batch_path else '❌'}")
    print(f"  - 步骤区: {'✅' if has_steps else '❌'}")
    print(f"  - 验证区: {'✅' if has_verification else '❌'}")

    # 检查是否有 from pathlib import Path
    print(f"\nPathlib 检查:")
    has_pathlib = False
    for cell in code_cells:
        source = ''.join(cell.get('source', []))
        if 'from pathlib import Path' in source:
            has_pathlib = True
            break

    print(f"  - 使用 from pathlib import Path: {'✅' if has_pathlib else '❌'}")

    # 检查错误处理
    print(f"\n错误处理检查:")
    has_try_except = False
    has_specific_error = False
    has_logger_error = False

    for cell in code_cells:
        source = ''.join(cell.get('source', []))
        if 'try:' in source:
            has_try_except = True
        if 'except Exception as e:' in source:
            has_specific_error = True
        if 'logger.error' in source:
            has_logger_error = True

    print(f"  - 使用 try/except: {'✅' if has_try_except else '❌'}")
    print(f"  - 使用具体异常: {'✅' if has_specific_error else '❌'}")
    print(f"  - 使用 logger.error: {'✅' if has_logger_error else '❌'}")

    # 检查函数长度
    print(f"\n函数长度检查:")
    large_functions = []
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        lines = source.split('\n')
        in_function = False
        function_name = ""
        function_lines = 0
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                if in_function and function_lines > 30:
                    large_functions.append(f"Cell {i+1}: {function_name} ({function_lines}行)")
                in_function = True
                function_name = line.strip().split('(')[0].replace('def ', '').replace('async def ', '')
                function_lines = 1
            elif in_function and line.strip() and not line.strip().startswith('#'):
                function_lines += 1
            elif in_function and line.strip() == '':
                function_lines += 1
            elif in_function and line.strip().startswith('#'):
                function_lines += 1
            elif in_function and (line.strip().startswith('def ') or line.strip().startswith('async def ')):
                # 新函数开始
                if function_lines > 30:
                    large_functions.append(f"Cell {i+1}: {function_name} ({function_lines}行)")
                function_name = line.strip().split('(')[0].replace('def ', '').replace('async def ', '')
                function_lines = 1
            else:
                in_function = False
                if function_lines > 30:
                    large_functions.append(f"Cell {i+1}: {function_name} ({function_lines}行)")
                function_lines = 0

        # 检查最后一个函数
        if in_function and function_lines > 30:
            large_functions.append(f"Cell {i+1}: {function_name} ({function_lines}行)")

    if large_functions:
        print(f"  发现 {len(large_functions)} 个超过30行的函数:")
        for func in large_functions[:5]:
            print(f"    {func}")
    else:
        print(f"  ✅ 未发现超过30行的函数")

    # 检查命名规范
    print(f"\n命名规范检查:")
    bad_names = ['data', 'info', 'tmp', 'res', 'result']
    bad_name_usage = []
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        for bad_name in bad_names:
            # 简单检查变量名
            if f' {bad_name} ' in source or f' {bad_name}=' in source or f'({bad_name},' in source:
                bad_name_usage.append(f"Cell {i+1}: 使用 '{bad_name}'")

    if bad_name_usage:
        print(f"  发现 {len(bad_name_usage)} 个可能的模糊命名:")
        for usage in bad_name_usage[:5]:
            print(f"    {usage}")
    else:
        print(f"  ✅ 未发现明显的模糊命名")

    return {
        'total_cells': len(cells),
        'markdown_cells': len(markdown_cells),
        'code_cells': len(code_cells),
        'has_logging': has_logging_import,
        'has_ipynbname': has_ipynbname_import,
        'has_logger': has_logger_creation,
        'has_log_file': has_log_file_handler,
        'log_file_exists': log_file.exists(),
        'has_display': has_display,
        'has_config': has_config,
        'has_batch_path': has_batch_path,
        'has_steps': has_steps,
        'has_verification': has_verification,
        'has_pathlib': has_pathlib,
        'has_try_except': has_try_except,
        'has_specific_error': has_specific_error,
        'has_logger_error': has_logger_error,
        'large_functions': len(large_functions),
        'bad_names': len(bad_name_usage),
        'print_issues': len(print_issues)
    }

def main():
    """主函数"""
    pipeline_dir = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData")

    # 查找所有 .ipynb 文件
    notebook_files = list(pipeline_dir.glob("*.ipynb"))

    if not notebook_files:
        print("❌ 未找到任何 .ipynb 文件")
        return

    print(f"找到 {len(notebook_files)} 个 Notebook 文件")
    print(f"{'='*80}")

    results = {}
    for nb_file in notebook_files:
        results[nb_file.name] = analyze_notebook(nb_file)

    # 生成总结报告
    print(f"\n{'='*80}")
    print("总结报告")
    print(f"{'='*80}")

    for filename, result in results.items():
        print(f"\n{filename}:")
        print(f"  总 cells: {result['total_cells']} (Markdown: {result['markdown_cells']}, Code: {result['code_cells']})")
        print(f"  日志配置: {'✅' if result['has_logging'] else '❌'} ipynbname: {'✅' if result['has_ipynbname'] else '❌'} logger: {'✅' if result['has_logger'] else '❌'}")
        print(f"  .log 文件: {'✅' if result['log_file_exists'] else '❌'}")
        print(f"  结构: 配置{'✅' if result['has_config'] else '❌'} 批量路径{'✅' if result['has_batch_path'] else '❌'} 步骤{'✅' if result['has_steps'] else '❌'} 验证{'✅' if result['has_verification'] else '❌'}")
        print(f"  其他: pathlib{'✅' if result['has_pathlib'] else '❌'} IPython.display{'✅' if result['has_display'] else '❌'}")
        print(f"  问题: 大函数{result['large_functions']} 模糊命名{result['bad_names']} print问题{result['print_issues']}")

if __name__ == "__main__":
    main()