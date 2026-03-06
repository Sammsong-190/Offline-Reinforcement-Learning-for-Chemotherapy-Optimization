"""Export offline_dataset.npz to CSV for viewing"""
import numpy as np
d = np.load('offline_dataset.npz')
s = d['s_raw'] if 's_raw' in d else d['s']
a_space = d.get('action_space', np.array([0., 0.5, 1., 2.]))
a = np.array([a_space[int(i)] for i in np.array(d['a']).flatten()])
r = d['r']
table = np.hstack([s, a.reshape(-1, 1), r.reshape(-1, 1)])
np.savetxt('offline_dataset.csv', table, delimiter=',',
           header='N,T,I,C,dose,reward', comments='')
print('Exported to offline_dataset.csv, rows:', len(table))
