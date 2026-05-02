import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         # print(cols)
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', stride=1):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         self.stride = stride
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         # print(cols)
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index * self.stride
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         n = len(self.data_x) - self.seq_len - self.pred_len + 1
#         if n <= 0:
#             return 0
#         return (n + self.stride - 1) // self.stride

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
    

import numpy as np
from utils.timefeatures import time_features


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='S',
        data_path='ETTh1.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        user_col='user_id',
        split_seed=42,
        stride=1,
        setting=None
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        self.flag = flag

        self.stride = stride
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.setting = setting
        
        self.user_col = user_col
        self.split_seed = split_seed

        self.__read_data__()

    # def _split_users(self, unique_users):
    #     """
    #     Split unique users into 80/10/10 with reproducibility.
    #     """
    #     unique_users = np.array(unique_users)
    #     rng = np.random.default_rng(self.split_seed)
    #     shuffled = unique_users.copy()
    #     rng.shuffle(shuffled)

    #     n_users = len(shuffled)
    #     if n_users < 3:
    #         raise ValueError(
    #             f"Need at least 3 unique users for train/val/test split, got {n_users}"
    #         )

    #     n_train = int(np.floor(0.8 * n_users))
    #     n_val = int(np.floor(0.1 * n_users))
    #     n_test = n_users - n_train - n_val

    #     # Make sure val/test are non-empty when possible
    #     if n_val == 0:
    #         n_val = 1
    #         n_train -= 1
    #     if n_test == 0:
    #         n_test = 1
    #         n_train -= 1

    #     # Safety
    #     if n_train <= 0:
    #         raise ValueError(
    #             f"Invalid split sizes after adjustment: "
    #             f"train={n_train}, val={n_val}, test={n_test}"
    #         )

    #     train_users = shuffled[:n_train]
    #     val_users = shuffled[n_train:n_train + n_val]
    #     test_users = shuffled[n_train + n_val:]
        
    #     return set(train_users), set(val_users), set(test_users)
    
    # def _split_users(self, unique_users):
    #     """
    #     Split unique users into:
    #     - 1 user for validation
    #     - 10% of the remaining users for testing
    #     - the rest for training

    #     Uses self.split_seed for reproducibility.
    #     """
    #     unique_users = np.array(unique_users)
    #     rng = np.random.default_rng(self.split_seed)
    #     shuffled = unique_users.copy()
    #     rng.shuffle(shuffled)

    #     n_users = len(shuffled)
    #     if n_users < 3:
    #         raise ValueError(
    #             f"Need at least 3 unique users for train/val/test split, got {n_users}"
    #         )

    #     # Fixed validation size
    #     n_val = 1

    #     # Remaining users after taking validation
    #     remaining = n_users - n_val

    #     # 10% of remaining for test
    #     n_test = int(np.floor(0.1 * remaining))

    #     # Ensure test is non-empty when possible
    #     if n_test == 0:
    #         n_test = 1

    #     n_train = n_users - n_val - n_test

    #     if n_train <= 0:
    #         raise ValueError(
    #             f"Invalid split sizes after adjustment: "
    #             f"train={n_train}, val={n_val}, test={n_test}"
    #         )

    #     val_users = shuffled[:n_val]
    #     test_users = shuffled[n_val:n_val + n_test]
    #     train_users = shuffled[n_val + n_test:]

    #     # print("-" * 50)
    #     # print("AutoFormer Dataset User Split:")
    #     # print(f"Total users: {n_users}, Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")
    #     # print(f"Train users: {train_users}")
    #     # print(f"Val users: {val_users}")
    #     # print(f"Test users: {test_users}")
    #     # print(f"Split seed: {self.split_seed}")
    #     # print("-" * 50)
        
    #     return set(train_users), set(val_users), set(test_users)
    
    def _split_users(self, unique_users):
        unique_users = np.sort(np.array(unique_users))
        rs = np.random.RandomState(self.split_seed)

        shuffled = unique_users.copy()
        rs.shuffle(shuffled)

        n_users = len(shuffled)
        if n_users < 3:
            raise ValueError(
                f"Need at least 3 unique users for train/val/test split, got {n_users}"
            )

        n_val = 1
        remaining = n_users - n_val
        n_test = max(1, int(np.floor(0.1 * remaining)))
        n_train = n_users - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                f"Invalid split sizes after adjustment: "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

        val_users = shuffled[:n_val]
        test_users = shuffled[n_val:n_val + n_test]
        train_users = shuffled[n_val + n_test:]

        # print("-" * 50)
        # print("AutoFormer Dataset User Split:")
        # print(f"Total users: {n_users}, Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")
        # print(f"Train users: {train_users}")
        # print(f"Val users: {val_users}")
        # print(f"Test users: {test_users}")
        # print(f"Split seed: {self.split_seed}")
        # print("-" * 50)
        
        return set(train_users), set(val_users), set(test_users)

    def _build_time_features(self, df_stamp):
        """
        Build time features exactly like the original repo.
        Expects df_stamp to contain a 'date' column only.
        """
        df_stamp = df_stamp.copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unknown timeenc: {self.timeenc}")

        return data_stamp

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        # could be csv or parquet, read accordingly
        if file_path.endswith('.csv'):
            df_raw = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df_raw = pd.read_parquet(file_path)
            df_raw.rename(columns={"time": "date"}, inplace=True)
            # currently, date is integer from 0 to len(df)-1. We convert to datetime to be compatible with time features. The actual date values do not matter as long as they are consistent and in order.
            sampling_rate = 50  # Hz
            start_time = pd.Timestamp("2020-01-01 00:00:00")
            df_raw['date'] = start_time + pd.to_timedelta(df_raw['date'] / sampling_rate, unit='s')
            df_raw.drop("label", axis=1, inplace=True)
            # save as csv with same name for consistency with rest of code
            csv_path = file_path[:-8] + '.csv'
            df_raw.to_csv(csv_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        if self.user_col not in df_raw.columns:
            raise ValueError(f"Expected user column '{self.user_col}' in dataset")
        if 'date' not in df_raw.columns:
            raise ValueError("Expected 'date' column in dataset")
        if self.target not in df_raw.columns:
            raise ValueError(f"Expected target column '{self.target}' in dataset")

        # Reorder columns so modeled columns remain:
        # ['date', user_id, feature1, ..., featureK, target]
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        cols.remove(self.user_col)
        df_raw = df_raw[['date', self.user_col] + cols + [self.target]]

        # Split users reproducibly
        unique_users = sorted(df_raw[self.user_col].unique())
        train_users, val_users, test_users = self._split_users(unique_users)

        if self.flag == 'train':
            selected_users = train_users
        elif self.flag == 'val':
            selected_users = val_users
        else:
            selected_users = test_users

        # Training dataframe for fitting scaler only
        df_train_all = df_raw[df_raw[self.user_col].isin(train_users)].copy()

        # Current split dataframe
        df_split = df_raw[df_raw[self.user_col].isin(selected_users)].copy()
        
        # Fit scaler on train users only, after dropping non-modeled columns
        df_train_model = df_train_all.drop(columns=[self.user_col]).copy()

        if self.features in ['M', 'MS']:
            train_cols_data = df_train_model.columns[1:]   # all except date
            train_data = df_train_model[train_cols_data].values
        elif self.features == 'S':
            train_data = df_train_model[[self.target]].values
        else:
            raise ValueError(f"Unknown features mode: {self.features}")

        # save Train data for using in evaluation
        if self.flag == 'train':
            # result save
            folder_path = './results/' + self.setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # enforce divisibility by seq_len + pred_len for simplicity
            n_train = len(train_data)
            n_train = (n_train // (self.seq_len + self.pred_len)) * (self.seq_len + self.pred_len)
            train_data_to_save = train_data[:n_train]
            # reshape to (num_windows, seq_len + pred_len, num_features)
            train_data_to_save = train_data_to_save.reshape(-1, self.seq_len + self.pred_len, train_data.shape[1])
            save_compressed_npz(data_file=train_data_to_save, model_name="autoformer", save_path=folder_path + 'TRAIN.npz')
            
        
        if self.scale:
            self.scaler.fit(train_data)

        # Store per-user arrays so windows do not cross users
        self.user_data_x = []
        self.user_data_y = []
        self.user_data_stamp = []
        self.index_map = []   # list of (user_block_idx, start_idx)

        grouped = df_split.groupby(self.user_col, sort=False)

        for user_value, user_df in grouped:
            user_df = user_df.copy()

            # Drop user_id after splitting
            user_df_model = user_df.drop(columns=[self.user_col]).reset_index(drop=True)

            # Select modeled columns
            if self.features in ['M', 'MS']:
                cols_data = user_df_model.columns[1:]   # all except date
                df_data = user_df_model[cols_data]
            else:  # 'S'
                df_data = user_df_model[[self.target]]

            data_values = df_data.values
            if self.scale:
                data_values = self.scaler.transform(data_values)

            df_stamp = user_df_model[['date']]
            data_stamp = self._build_time_features(df_stamp)

            # Save this user's arrays
            block_idx = len(self.user_data_x)
            self.user_data_x.append(data_values)
            self.user_data_y.append(data_values)
            self.user_data_stamp.append(data_stamp)

            # Build valid start indices for this user only
            user_len = len(data_values)
            n_windows = user_len - self.seq_len - self.pred_len + 1

            # control stride based on flag. For training, use self.stride. For val/test, use stride=256 as other baselines
            if self.flag == 'train':
                stride = self.stride
            elif self.flag == 'val':
                stride = 32
            else:
                stride = 256
            if n_windows > 0:
                for start_idx in range(0, n_windows, stride):
                    self.index_map.append((block_idx, start_idx))

        if len(self.index_map) == 0:
            raise ValueError(
                f"No valid windows found for split='{self.flag}'. "
                f"Check seq_len={self.seq_len}, pred_len={self.pred_len}, "
                f"and per-user sequence lengths."
            )

    def __getitem__(self, index):
        block_idx, s_begin = self.index_map[index]

        data_x = self.user_data_x[block_idx]
        data_y = self.user_data_y[block_idx]
        data_stamp = self.user_data_stamp[block_idx]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.index_map)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def save_compressed_npz(
        data_file,
        channel_names=None,
        model_name="par",
        save_path="generated_samples",
        seed=42,
    ):
        """
        Save one dataset already formatted as [N, T, C] into a compressed .npz file.

        Parameters
        ----------
        data_file : np.ndarray
            Array of shape [N, T, C].
        channel_names : list[str] | None
            Optional list of channel names of length C.
            If None, default names channel_0 ... channel_{C-1} are used.
        model_name : str
            Name of the model.
        save_path : str
            Output path, with or without '.npz'.
        seed : int
            Seed metadata to store in the file.
        """
        samples = np.asarray(data_file, dtype=np.float32)

        if samples.ndim != 3:
            raise ValueError(f"`data_file` must have shape [N, T, C], got {samples.shape}")

        N, seq_len, num_channels = samples.shape

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(num_channels)]
        else:
            if len(channel_names) != num_channels:
                raise ValueError(
                    f"len(channel_names) must equal num_channels={num_channels}, "
                    f"got {len(channel_names)}"
                )

        if not save_path.endswith(".npz"):
            save_path = f"{save_path}.npz"

        np.savez_compressed(
            save_path,
            samples=samples,
            channel_names=np.array(channel_names, dtype=object),
            seq_len=np.int32(seq_len),
            num_channels=np.int32(num_channels),
            num_samples=np.int32(N),
            model_name=model_name,
            seed=np.int32(seed),
        )
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
