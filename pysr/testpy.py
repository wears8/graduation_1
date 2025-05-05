import os
import pandas as pd
import numpy as np
import pysindy as ps
from update_expressions import update_expressions

# 自定义指数函数库
def exp_function(x):
    return np.exp(x)

# 构建自定义特征库
exp_library = ps.CustomLibrary(
    library_functions=[exp_function],
    function_names=['exp(x)']
)

# 示例：将指数函数库与其他库组合
poly_lib = ps.PolynomialLibrary(degree=3)  # 多项式库
combined_library = ps.GeneralizedLibrary([poly_lib, exp_library])

# 构建 SINDy 模型
model = ps.SINDy(feature_library=combined_library)

# 示例数据
t = np.linspace(0, 10, 100)
x = np.sin(t).reshape(-1, 1)

# 拟合模型
model.fit(x, t=t[1] - t[0])
model.print()

# 读取并准备数据
df = pd.read_csv('./train.csv')
data = np.array(df)
X = data[:, :-1]
y = data[:, -1].reshape(-1)

# 构建数据字典
dataset = {
    'data': {
        'inputs': X,
        'outputs': y
    }
}

# 使用update_expressions函数处理数据并更新文件
update_expressions('specification_oscillator1_body.txt', dataset)

print("处理完成！请检查 specification_oscillator1_body.txt 文件")