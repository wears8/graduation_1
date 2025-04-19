import re
import numpy as np
import sympy
from pysr import PySRRegressor
from typing import List, Tuple, Dict, Optional, Union
import shutil

class ExpressionUpdater:
    def __init__(self, spec_file_path: str, dataset: Optional[Dict] = None):
        """初始化表达式更新器
        
        Args:
            spec_file_path: specification文件的路径
            dataset: 可选的数据集字典，包含 'data' 键，其中包含 'inputs' 和 'outputs'
        """
        self.spec_file_path = spec_file_path
        self.var_mapping = None
        self.math_prefix = None
        self.dataset = dataset
        self._init_var_mapping()
    
    def _init_var_mapping(self):
        """从specification文件中初始化变量映射和数学库前缀"""
        with open(self.spec_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 找到equation函数的参数定义
        match = re.search(r'def equation\((.*?)\)', content)
        if not match:
            raise ValueError("无法在specification文件中找到equation函数定义")
        
        # 提取参数及其类型
        param_defs = [p.strip() for p in match.group(1).split(',') if 'params' not in p]
        if not param_defs:
            raise ValueError("未找到有效的变量参数")

        # 创建变量映射和收集所有类型
        self.var_mapping = {}
        type_modules = set()
        
        for i, param in enumerate(param_defs):
            name_type = param.split(':')
            if len(name_type) != 2:
                continue
            
            name, type_hint = name_type
            name = name.strip()
            self.var_mapping[f'x{i}'] = name
            
            # 提取模块名（np, jnp, torch等）
            type_match = re.search(r'(\w+)\.', type_hint)
            if type_match:
                type_modules.add(type_match.group(1))
        
        # 根据类型优先级选择数学库前缀
        if 'torch' in type_modules:
            self.math_prefix = 'torch.'
        elif 'jnp' in type_modules:
            self.math_prefix = 'jnp.'
        else:
            self.math_prefix = 'np.'
    
    def _replace_math_functions(self, expr: str) -> str:
        """替换数学函数为指定库的版本"""
        # 定义需要替换的数学函数
        math_funcs = ['sin', 'cos', 'log', 'exp']
        
        # 替换所有未带前缀的数学函数
        for func in math_funcs:
            pattern = r'(?<!\.)(?<!\w)' + func + r'(?!\w)'
            expr = re.sub(pattern, f'{self.math_prefix}{func}', expr)
        
        return expr
    
    def _replace_constants_with_params(self, expr: str) -> str:
        """将表达式中的常数替换为params数组项"""
        param_counter = 0
        
        def replace_with_param(match):
            nonlocal param_counter
            current = param_counter
            param_counter += 1
            return f"params[{current}]"
        
        # 匹配所有数字（排除科学记数法中的指数）
        expr = re.sub(r'(?<![e|E])-?\d+\.?\d*(?![e|E]\d+)', replace_with_param, expr)
        
        return expr
    
    def _clean_file_content(self) -> str:
        """读取并清理文件内容，移除已存在的表达式块"""
        with open(self.spec_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割内容以找到所有的 INITIAL BODIES 块
        parts = content.split("# === INITIAL BODIES START ===")
        
        if len(parts) > 1:
            # 保留第一部分（主要内容）
            return parts[0].rstrip()
        else:
            return content.rstrip()
    
    def process_expressions(self, model: PySRRegressor, start_idx: int = 2, top_n: int = 5) -> List[str]:
        """处理模型生成的表达式
        
        Args:
            model: 训练好的PySR模型
            start_idx: 开始提取的索引（从0开始，默认为2表示第3个）
            top_n: 要处理的表达式数量（默认为5）
        
        Returns:
            处理后的表达式列表
        """
        # 按score降序排序（score越高越好）
        sorted_equations = model.equations_.sort_values('score', ascending=False)
        processed_expressions = []
        
        # 计算结束索引
        end_idx = min(start_idx + top_n, len(sorted_equations))
        
        # 提取指定范围的表达式
        for i in range(start_idx, end_idx):
            expr = sorted_equations.iloc[i]
            sympy_expr = str(expr['sympy_format'])
            
            # 1. 转换为sympy表达式并简化
            sympy_expr = sympy.sympify(sympy_expr)
            sympy_expr = sympy.simplify(sympy_expr)
            sympy_expr = str(sympy_expr)
            
            # 2. 替换变量名
            for old_var, new_var in self.var_mapping.items():
                sympy_expr = sympy_expr.replace(old_var, new_var)
            
            # 3. 替换数学函数
            sympy_expr = self._replace_math_functions(sympy_expr)
            
            # 4. 替换常数为params数组项
            sympy_expr = self._replace_constants_with_params(sympy_expr)
            
            # 构建完整的函数体
            body = f"dv = {sympy_expr}\nreturn dv"
            processed_expressions.append(body)
        
        return processed_expressions
    
    def update_file(self, model: PySRRegressor, start_idx: int = 2, top_n: int = 5) -> None:
        """更新文件中的表达式
        
        Args:
            model: 训练好的PySR模型
            start_idx: 开始提取的索引（从0开始，默认为2表示第3个）
            top_n: 要处理的最佳表达式数量（默认为5）
        """
        # 获取处理后的表达式
        processed_expressions = self.process_expressions(model, start_idx, top_n)
        
        # 清理并准备文件内容
        base_content = self._clean_file_content()
        
        # 构建新内容
        new_content = [
            base_content,  # 原始内容
            "\n",  # 空行
            "# === INITIAL BODIES START ===" 
        ]
        
        # 添加每个表达式
        for expr in processed_expressions:
            new_content.extend([
                "",
                "# --- BODY START ---",
                expr,
                "# --- BODY END ---"
            ])
        
        # 添加结束标记
        new_content.extend([
            "",
            "# === INITIAL BODIES END ==="
        ])
        
        # 写回文件
        with open(self.spec_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_content))
    
    def process_dataset(self):
        """处理数据集并更新文件"""
        if self.dataset is None:
            raise ValueError("没有提供数据集")
            
        data = self.dataset['data']
        X = data['inputs']
        y = data['outputs']
        
        # 配置 PySR
        model = PySRRegressor(
            temp_equation_file=False,  # 不保存方程式文件
            delete_tempfiles=True,  # 删除临时文件
            niterations=40,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=15,
            population_size=30,
            verbosity=1,
            progress=True,
            procs=4,
            parsimony=0.1
        )
        
        # 训练模型
        print("开始训练符号回归模型...")
        model.fit(X, y)
        print("模型训练完成！")

        # 更新文件
        self.update_file(model)

        shutil.rmtree("outputs", ignore_errors=True)
        shutil.rmtree("pysr_ws", ignore_errors=True)
        
        
def update_expressions(spec_file_path: str, model: Optional[Union[PySRRegressor, Dict]] = None, top_n: int = 5) -> None:
    """更新指定文件中的表达式
    
    Args:
        spec_file_path: specification文件的路径
        model: 训练好的PySR模型或包含数据的字典
        top_n: 要处理的最佳表达式数量（默认为5）
    """
    updater = ExpressionUpdater(spec_file_path)
    if isinstance(model, dict):
        # 如果提供的是数据集，手动调用处理函数
        updater.dataset = model
        updater.process_dataset()
    else:
        # 如果提供的是训练好的模型，直接更新文件
        updater.update_file(model, top_n)