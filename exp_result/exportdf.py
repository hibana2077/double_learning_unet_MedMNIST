import json
import os
import pandas as pd

TARGET_DIR = './'

# 取得目標目錄中的資料夾名稱
dir_list = [d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d))]

# 建立一個空的 DataFrame
df = pd.DataFrame()

# 遍歷每個資料夾
for dataset in dir_list:
    dataset_path = os.path.join(TARGET_DIR, dataset)
    print(f'Processing {dataset}...')
    # print(os.listdir(dataset_path))
    # 遍歷資料夾中的每個檔案
    for file in os.listdir(dataset_path):
        if file.endswith('.json'):
            file_path = os.path.join(dataset_path, file)
            # print(f'Processing {file_path}...')
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(f"model {file.split('.')[0]}: {max(data['test_acc'])} in {dataset}")
                # df[file.split('.')[0], dataset] = 
                df.loc[file.split('.')[0], dataset] = max(data['test_acc'])

# 將 DataFrame 轉換成 JSON
print(df)

# 將 DataFrame 寫入 JSON 檔案
df.to_json('result.json', orient='index')