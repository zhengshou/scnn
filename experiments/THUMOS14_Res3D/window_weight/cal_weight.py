# ----------------------------------------------------------------------------------------------------------------
# Segment-CNN
# Copyright (c) 2016 , Digital Video & Multimedia (DVMM) Laboratory at Columbia University in the City of New York.
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheng Shou, Dongang Wang, and Shih-Fu Chang.
# ----------------------------------------------------------------------------------------------------------------

import csv

# thumos 20 classes
classname = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling',
'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 
'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']

filepath = '../annotation/annotation_val/'
window_length = [0.6, 1.24, 2.52, 5.08, 10.20, 22.44]
window_cnt = [[0]*6 for _ in range(20)]
weight = []

for i in range(20):
    f = open(filepath+classname[i]+'_val.txt', 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        delta = float(line[3])-float(line[2])
        for j in range(6):
            if delta>=(window_length[j]/2) and delta <= (window_length[j]*2):
                window_cnt[i][j] = window_cnt[i][j]+1
    weight.append([float(w)/sum(window_cnt[i]) for w in window_cnt[i]])

# transpose for further usage
weight = [[row[i] for row in weight] for i in range(6)]

# write to csv
myfile = open('weight.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerows(weight)
myfile.close()