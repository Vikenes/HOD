import numpy as np
import pandas as pd


DATA_DIR = '/cosma7/data/dp004/dc-cues1/DarkQuest/linear/'
cosmo3d = np.loadtxt(DATA_DIR + "cosmo.dat")
logtk = np.load(DATA_DIR + "logtklist.npy")

features = {
            'wb': cosmo3d[:,0],
            'wc': cosmo3d[:,1],
            'Ode': cosmo3d[:,2],
}
for k in range(logtk.shape[1]):
    features[f'pk_{k}'] = logtk[:,k]

df = pd.DataFrame(
        features
)
all_idx = list(range(len(df)))
test_idx = np.random.choice(all_idx, size=int(0.1*(len(df))),replace=False)
val_idx = np.random.choice([idx for idx in all_idx if idx not in test_idx], size=int(0.1*len(df)),replace=False,)
combo = list(test_idx) + list(val_idx)
train_idx = [idx for idx in all_idx if idx not in combo]

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]
test_df = df.iloc[test_idx]

test_df.to_csv(DATA_DIR + 'test_linear_multi.csv',index=False,)
train_df.to_csv(DATA_DIR + 'train_linear_multi.csv',index=False,)
val_df.to_csv(DATA_DIR + 'val_linear_multi.csv',index=False,)
