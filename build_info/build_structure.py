import os

# 定义目录结构
def convert_to_structure(content: str):
    """
    将输入的目录结构文本转换为 Python 字典格式。
    :param content: 目录结构的文本描述
    :return: 转换后的字典结构
    """
    # 用于存储转换后的目录结构
    dir_structure = {}

    # 分割输入内容为每行
    lines = content.strip().split('\n')

    # 递归函数，构建嵌套目录结构
    def parse_lines(lines, level=0):
        nonlocal dir_structure
        current_dir = {}
        indent = '    '  # 设定缩进的空格数

        while lines:
            line = lines.pop(0)

            # 判断当前行的缩进级别
            current_level = line.find(line.strip())

            if current_level < level:
                # 如果当前行缩进级别小于当前目录深度，说明是上层目录的内容
                lines.insert(0, line)  # 退回当前行
                break

            # 获取文件名和注释内容
            line = line.strip()
            if line.endswith('/'):
                # 如果是目录
                dir_name = line[:-1]  # 去掉末尾的 '/'
                current_dir[dir_name] = parse_lines(lines, current_level + 1)
            elif '#' in line:
                # 如果是文件并且包含注释
                file_name, comment = line.split('#', 1)
                current_dir[file_name.strip()] = f"# {comment.strip()}"
            else:
                # 处理没有注释的文件（如果有的话）
                current_dir[line.strip()] = ''

        return current_dir

    # 开始解析传入的内容
    dir_structure = parse_lines(lines)

    return dir_structure

# 创建目录和文件
def create_structure(base_path, structure):
    for name, value in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(value, dict):
            # 创建文件夹
            os.makedirs(path, exist_ok=True)
            # 递归创建子目录
            create_structure(path, value)
        elif isinstance(value, str):
            # 创建文件并写入注释
            with open(path, 'w', encoding='utf-8') as f:
                f.write(value)
        else:
            print(f"Unexpected value type for {name}")

# 设置根目录路径
# root_path = 'project_structure'

# 创建根目录
# os.makedirs(root_path, exist_ok=True)

root_path = 'D:/Objects/Nebula_Model/models'

structure = {
    'inference': {
        '__init__.py': "# 推理模块初始化",
        'infer.py': "# 推理主程序，负责推理流程的启动和管理",
        'beam_search.py': "# 束搜索解码策略，优化生成过程，加速推理",
        'decoding.py': "# 解码策略，支持多种生成方式（贪心、随机采样、Top-k、Top-p等）",
        'postprocessing.py': "# 生成结果的后处理（格式化、去重、拼接等）",
        'serving.py': "# 部署推理服务（API接口，如Flask/FastAPI等）",
        'utils.py': "# 推理相关的工具函数（模型加载、输入输出处理、数据转换等）",
        'config.py': "# 配置文件，存储推理相关的超参数、路径设置等",
        'model_manager.py': "# 管理多个模型加载、切换、版本控制等",
        'optimization': {
            '__init__.py': "# 优化模块初始化",
            'quantization.py': "# 量化模块，优化模型大小和推理速度",
            'pruning.py': "# 剪枝模块，优化模型结构，提高推理效率",
            'tensor_optimization.py': "# 张量优化模块，提升推理过程中的计算效率"
        }
    }
}


# 创建目录结构
create_structure(root_path, structure)

print(f"项目目录结构已创建: {root_path}")
