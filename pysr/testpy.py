import os
import pandas as pd
import numpy as np
from update_expressions import update_expressions

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