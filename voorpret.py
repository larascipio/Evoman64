
    # imports libraries 
import pandas as pd
from csv import reader, writer
import matplotlib.pyplot as plt
    
# df1 = pd.read_csv('Average_8.csv')
# df1.insert(0, 'Enemy', 8)
# df1.insert(0, 'Exp_name', 'neat')
# df1.to_csv('neat_8_results.csv', index=False)

df = pd.concat(map(pd.read_csv, ['neat_3_results.csv', 'neat_6_results.csv', 'neat_8_results.csv']), ignore_index=True)
# df2 = pd.read_csv('ea10_best2.csv')
# df_merged = df1.append(df2, ignore_index=True)
# df_merged2 = df_merged.append()
# df = pd.read_csv('ea10_results.csv')
df.to_csv('neat_results.csv', index=False)
