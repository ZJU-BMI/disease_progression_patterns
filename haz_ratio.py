import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


data_name = 'mcitoad'
df = pd.read_csv('result_adni_{}2/result/cluster_res_{}.csv'.format(data_name, data_name))
# labels = df.columns[3:].tolist()
labels = ['AGE', 'PTGENDER', 'APOE4', 'MMSE', 'Subtype']
df_res = pd.DataFrame({
    'labels': labels,
    'haz': 0.,
    'lower': 0.,
    'upper': 0,
    'pval': 0.,
    'cindex': 0,
})

haz_res = []
ci = np.zeros([len(labels), 2])
pval = []
cindex = []

for i, label in enumerate(labels):
    df2 = df[['time', 'label', label]]
    df_x = df2[df2['time'] == 0]
    df_y = df2[df2['time'] > 0]
    df_data = df_x.copy()
    df_data.loc[:, 'time'] = df_y['time'].values
    df_data.loc[:, 'label'] = df_y['label'].values

    cph = CoxPHFitter()
    cph.fit(df_data, 'time', 'label')
    haz_res.append(cph.hazard_ratios_.values[0])
    ci[i, :] = np.exp(cph.confidence_intervals_.values)
    cindex.append(cph.concordance_index_)
    pval.append(cph.summary['p'].tolist()[0])

    print(label + ' hazard ratio: {:.3f}({:.3f}-{:.3f})'.format(haz_res[i], ci[i, 0], ci[i, 1]))


df_res['haz'] = haz_res
df_res['lower'] = ci[:, 0]
df_res['upper'] = ci[:, 1]
df_res['pval'] = pval
df_res['cindex'] = cindex

# df_res = df_res[(df_res['pval'] < 0.05) & (~df_res['haz'].between(0.98, 1.02))]
# df_res.sort_values(by=['cindex'], ascending=False, inplace=True)
df_res.to_csv('haz_ratio/haz_ratio_{}.csv'.format(data_name), index=False)
