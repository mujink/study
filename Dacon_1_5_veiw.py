import pandas as pd
import numpy as np

submission = pd.read_csv('../data/csv/submission_v4.csv')
# ?????

print(submission['q_0.1'])
print(type(submission))

submission = np.array(submission)
# print(submission[0])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,3.7))
plt.subplot(3,1,1)
plt.plot(submission, label='y_predict')
plt.legend()
# plt.figure(figsize=(12, 8))
# # plt.plot(submission['q_0.1'].history)
# # plt.plot(submission.history['q_0.2'])
# # plt.plot(submission['q_0.3'])
# # plt.plot(submission['q_0.4'])
# # plt.plot(submission['q_0.5'])
# # plt.plot(submission['q_0.6'])
# # plt.plot(submission['q_0.7'])
# # plt.plot(submission['q_0.8'])
# # plt.plot(submission['q_0.9'])
# submission.plot()

# plt.title('submission (q)')
# plt.ylabel('TARGET')
# plt.xlabel('Time[30]')
# plt.legend(['q_0.1','q_0.2','q_0.3','q_0.4','q_0.5','q_0.6','q_0.7','q_0.8','q_0.9'])
# plt.show()





# plot_col_index = self.column_indices[plot_col]

# plt.subplot(3, 1, n+1)
# plt.ylabel(f'{plot_col} [normed]')
# plt.plot(self.input_indices, inputs[n, :, plot_col_index],label='Inputs', marker='.',zorder=-10)
# plt.scatter(self.label_indices, labels[n, :, label_col_index],edgecolors='k', label='Labels', c='#2ca02c', s=20)
# plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='none', label=f'Predictions(q={quantile})', s=15)
# plt.legend()
# plt.xlabel('Time [30m]')

# WindowGenerator.quantile_plot = quantile_plot



# w1.quantile_plot(model, quantile=q)

# def quantile_plot(self, model=None, plot_col='TARGET', max_subplots=3, quantile=None):
#       inputs, labels = self.example
#   if quantile == 0.1:
#     plt.figure(figsize=(12, 8))
#   plot_col_index = self.column_indices[plot_col]
#   max_n = min(max_subplots, len(inputs))
#   for n in range(max_n):
#     plt.subplot(3, 1, n+1)
#     plt.ylabel(f'{plot_col} [normed]')
#     if quantile == 0.1:
#       plt.plot(self.input_indices, inputs[n, :, plot_col_index],
#               label='Inputs', marker='.',zorder=-10)
#     if self.label_columns:
#       label_col_index = self.label_columns_indices.get(plot_col, None)
#     else:
#       label_col_index = plot_col_index
#     if label_col_index is None:
#       continue
#     if quantile == 0.1:
#       plt.scatter(self.label_indices, labels[n, :, label_col_index],
#                 edgecolors='k', label='Labels', c='#2ca02c', s=20)
#     if model is not None:
#       predictions = model(inputs)
#       plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='none', label=f'Predictions(q={quantile})', s=15)
#     if quantile == 0.9 and n==0:
#       plt.legend()
#   plt.xlabel('Time [30m]')

# WindowGenerator.quantile_plot = quantile_plot