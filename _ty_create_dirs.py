#!/usr/bin/env python3

import yaml
from pathlib import Path
from collections import defaultdict

def create_directories_from_yaml(yaml_file):
    """根据YAML配置创建目录结构"""
    
    # 读取YAML配置
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_dir = Path(config.get('base_dir', '.'))
    structure = config.get('structure', [])
    files = config.get('files', [])
    
    print("📁 开始创建YOLO项目目录结构...")
    print(f"📍 基础目录: {base_dir.absolute()}")
    print("=" * 50)
    
    # 创建目录
    created_dirs = []
    for dir_path in structure:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(dir_path)
        print(f"✅ 创建目录: {dir_path}")
    
    # 创建空文件
    created_files = []
    for file_path in files:
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch(exist_ok=True)
        created_files.append(file_path)
        print(f"📄 创建文件: {file_path}")
    
    print("=" * 50)
    print(f"🎉 目录结构创建完成！")
    
    # 显示树状结构
    display_tree_structure(base_dir, structure, files)
    
    return created_dirs, created_files

def display_tree_structure(base_dir, dirs, files):
    """显示树状目录结构"""
    
    print(f"\n🌳 生成的目录结构:")
    
    # 构建完整的树结构
    tree = defaultdict(list)
    all_paths = []
    
    # 添加 base_dir 下的所有路径
    for path in dirs + files:
        full_path = f"{base_dir.name}/{path}" if base_dir.name != "." else path
        all_paths.append(full_path)
    
    for path in sorted(all_paths):
        parts = path.split('/')
        for i in range(1, len(parts)):
            parent = '/'.join(parts[:i])
            child = '/'.join(parts[:i+1])
            if child not in tree[parent]:
                tree[parent].append(child)
    
    # 递归打印树状结构
    def print_tree(node, prefix="", is_last=True):
        if node:
            name = node.split('/')[-1] if '/' in node else node
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{name}")
            
            children = sorted(tree[node])
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            for i, child in enumerate(children):
                print_tree(child, new_prefix, i == len(children) - 1)
    
    # 从项目根目录开始打印
    if base_dir.name != ".":
        print(f"{base_dir.name}/")
        root_items = sorted(tree[base_dir.name])
        for i, item in enumerate(root_items):
            print_tree(item, "", i == len(root_items) - 1)
    else:
        print(".")
        root_items = sorted(tree[""])
        for i, item in enumerate(root_items):
            print_tree(item, "", i == len(root_items) - 1)
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"   目录数量: {len(dirs)}")
    print(f"   文件数量: {len(files)}")
    print(f"   总计: {len(dirs) + len(files)} 个项目")

def main():
    try:
        created_dirs, created_files = create_directories_from_yaml("_ty_dir_structure.yaml")
        
        print(f"\n💡 使用说明:")
        print(f"   修改 dir_structure.yaml 可以调整目录结构")
        print(f"   重新运行此脚本会更新目录结构")
        
    except FileNotFoundError:
        print("❌ 错误: 找不到 dir_structure.yaml 文件")
        print("💡 请确保 YAML 配置文件存在")
    except yaml.YAMLError as e:
        print(f"❌ YAML 解析错误: {e}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()