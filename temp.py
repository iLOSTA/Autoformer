import pandas as pd

df = pd.read_parquet("data\PPG_activity_7_all_raw_data_for_other_baselines.parquet")
# for each user in 'chest_subject' we add a 'time' column which is just integer from 0 to max len
df['time'] = df.groupby('chest_subject').cumcount()
print(df.tail(50))
# save to the same parquet file
df.to_parquet("data\PPG_activity_7_all_raw_data_for_other_baselines.parquet", index=False)