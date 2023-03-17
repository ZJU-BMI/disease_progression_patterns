import os.path
import pandas as pd
import numpy as np
import torch

import utils
from Staging_model import StagingModel

from import_longitudinal_data import import_data_adni

data_names = ['cntomci', 'mcitoad']

dataname = data_names[1]
data_path = 'dataset/ADNI_{}_data.csv'.format(dataname)
(x_dim, max_len), (x, y), \
    (time, label, attn_mask, time_seq) = import_data_adni(data_name=dataname,
                                                          normalize=True,
                                                          normal_mode='standard',
                                                          data_path=None)

save_path = 'result_adni_{}k33'.format(dataname)
res_path = save_path + '/result'
if not os.path.exists(res_path):
    os.makedirs(res_path)
print('num features:', x.shape[2])
print('max times:', np.max(time))

batch_size = 32
epochs = 50
pre_train_epochs = 40
lr = 3e-5
pre_lr = 3e-5
nClusters = 3
replace = True

input_dims = {
    'x_dim': x_dim,
    'dim': 256,
    'n_heads': 3,
    'dim_fc': 256,
    'dim_head': 64,
    'depth': 2,
    'max_len': max_len,
    'pool': 'mean',
    'dropout': 0.2,
    'decoder_dim1': 200,
    'decoder_dim2': 100,
    'decoder_dim3': 50,
    'z_dim': 3,
    'nClusters': nClusters,
    'cs_dim1': 200,
    'cs_dim2': 200,
    'num_times': int(np.max(time) * 1.1),
}

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = StagingModel(x, y, time, label, attn_mask, time_seq, input_dims=input_dims, nClusters=nClusters,
                     epochs=epochs, batch_size=batch_size, lr=lr, save_path=save_path)
model.fit(pre_train=True, pre_train_epochs=pre_train_epochs, pre_lr=pre_lr, replace=replace)
model.evaluate()

clusters = model.get_pred_cluster()
# pred_risks
print('predicting survival risk...')
pred_risk = model.pred_survival(x, attn_mask=attn_mask, time_seq=time_seq)
cumu_risk = np.cumsum(pred_risk, axis=1)

# survival plot
utils.survival_plot(time, label, clusters, save_path=res_path)
# save results
print('saving results....')
df2 = pd.read_csv('dataset/ADNI_{}_data.csv'.format(dataname))
df2['Subtype'] = np.repeat(np.reshape(clusters, [-1, 1]), 2, axis=1).reshape([-1])
df2.to_csv(res_path + '/cluster_res_{}k33.csv'.format(dataname), index=False)

# save risks
df_risk = pd.DataFrame(cumu_risk)
df_risk['time'] = time
df_risk['label'] = label
df_risk.to_csv(res_path + '/pred_risk_{}.csv'.format(dataname), index=False)

# attn_scores for each subtype
# attn_scores = model.get_attn_score(x)
# attn_scores_subtype1 = attn_scores[np.where(clusters == 0)[0]]
# attn_scores_subtype2 = attn_scores[np.where(clusters == 1)[0]]
# attn_scores_subtype1 = np.mean(attn_scores_subtype1, axis=0)
# attn_scores_subtype2 = np.mean(attn_scores_subtype2, axis=0)

# z
# z = model.get_z()
# h = model.get_deep_feature()
# # np.save(save_path + '/result/z.npy', z)
# np.save(save_path + '/result/h.npy', h)