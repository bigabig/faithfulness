from pathlib import Path
import pandas as pd

dataset = "summeval"
out_file = Path(f"{dataset}.csv")
folder_path = Path(dataset)
csv_files = folder_path.glob('*/*.csv')

df = pd.DataFrame()
for csv_file in csv_files:

    if csv_file.name.endswith("binary.csv"):
        continue

    df = pd.concat([df, pd.read_csv(csv_file)], ignore_index=True)

df['Variant'] = df.apply(lambda x: x['Method'].split('_')[2] if len(x['Method'].split('_')) == 3 else x['Method'].split('_')[3], axis=1)
df['Metric'] = df.apply(lambda x: x['Method'].split('_')[1] if len(x['Method'].split('_')) == 3 else x['Method'].split('_')[2], axis=1)
df['Method'] = df.apply(lambda x: x['Method'].split('_')[0] if len(x['Method'].split('_')) == 3 else x['Method'].split('_')[0] + x['Method'].split('_')[1], axis=1)
df['Method'] = df.apply(lambda x: x['Method'].split('_')[0], axis=1)

df.to_csv(out_file.name, index=False, encoding='UTF-8', sep=',')
