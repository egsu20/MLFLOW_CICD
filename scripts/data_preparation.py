import pandas as pd
from sklearn.datasets import load_boston

# 데이터 준비
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target
df.to_csv('data/boston.csv', index=False)
