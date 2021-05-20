import pandas as pd
import numpy as np
# ================================
# i = 0
# test = pd.read_csv('../data/csv/Dacon/train/train.csv')
# ranges = 336
# hours = range(ranges)

# test = test.iloc[ranges*(i):ranges*(i+1)]

# #  Hour       GHI      T-Td        Td    TARGET       DHI       DNI        WS        RH         T   Target1   Target2
# Hour = test['Hour'].values
# GHI = test['GHI'].values
# T_Td = test['T-Td'].values
# Td = test['Td'].values
# TARGET = test['TARGET'].values
# DHI = test['DHI'].values
# DNI = test['DNI'].values
# WS = test['WS'].values
# RH = test['RH'].values
# T = test['T'].values
# TARGET1 = test['Target1'].values
# TARGET2 = test['Target2'].values
# # print(type(Hour))
# # Hour = Hour[:range]
# # GHI = GHI[:range]
# # T_Td = T_Td[:range]
# # Td = Td[:range]
# # TARGET = TARGET[:range]
# # DHI = DHI[:range]
# # DNI = DNI[:range]
# # WS = WS[:range]
# # RH = RH[:range]
# # T = T[:range]
# # TARGET1 = TARGET1[:range]
# # TARGET2 = TARGET2[:range]


# import matplotlib.pyplot as plt
# plt.figure(figsize=(18,8))
# plt.subplot(6,1,1)
# # plt.plot(hours, Hour, color='red')
# plt.plot(hours, GHI, color='green')
# plt.subplot(6,1,2)
# plt.plot(hours, TARGET, color='blue')
# # plt.plot(hours, T_Td, color='#ddff00')
# plt.subplot(6,1,3)
# plt.plot(hours, TARGET1, color='#aacc00')
# # plt.plot(hours, Td, color='#886611')
# plt.subplot(6,1,4)
# plt.plot(hours, TARGET2, color='#aaccff')
# plt.subplot(6,1,5)
# plt.plot(hours, DHI, color='#ffaacc')
# plt.subplot(6,1,6)
# plt.plot(hours, DNI, color='red')
# # plt.plot(hours, WS, color='yellow')
# # plt.plot(hours, RH, color='blue')
# # plt.plot(hours, T, color='green')
# plt.legend()
# plt.show()

# ================================
#  나쁘지 않아!!
submission_v4 = pd.read_csv('../data/csv/submission_v4_2.csv')

ranges = 336
hours = range(ranges)
i = 6
submission_v4 = submission_v4[ranges*(i):ranges*(i+1)]

q_01 = submission_v4['q_0.1'].values
q_02 = submission_v4['q_0.2'].values
q_03 = submission_v4['q_0.3'].values
q_04 = submission_v4['q_0.4'].values
q_05 = submission_v4['q_0.5'].values
q_06 = submission_v4['q_0.6'].values
q_07 = submission_v4['q_0.7'].values
q_08 = submission_v4['q_0.8'].values
q_09 = submission_v4['q_0.9'].values

import matplotlib.pyplot as plt
plt.figure(figsize=(18,2.5))
plt.subplot(1,1,1)
plt.plot(hours, q_01, color='red')
plt.plot(hours, q_02, color='orange')
plt.plot(hours, q_03, color='yellow')
plt.plot(hours, q_04, color='green')
plt.plot(hours, q_05, color='blue')
plt.plot(hours, q_06, color='indigo')
plt.plot(hours, q_07, color='purple')
plt.plot(hours, q_08, color='black')
plt.plot(hours, q_09, color='brown')
plt.legend()
plt.grid(True)
plt.show()
