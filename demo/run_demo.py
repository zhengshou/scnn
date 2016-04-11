# ----------------------------------------------------------------------------------------------------------------
# Segment-CNN
# Copyright (c) 2016 , Digital Video & Multimedia (DVMM) Laboratory at Columbia University in the City of New York.
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheng Shou, Dongang Wang, and Shih-Fu Chang.
# ----------------------------------------------------------------------------------------------------------------

import sys, os
import shutil
import csv
import struct
import operator
import time
from argparse import ArgumentParser
from math import log

def main():
    print 'example: python run_demo.py -i video/video_test_0000131.mp4 -f 25'
	
    parser = ArgumentParser(description='SCNN demo, input the video\'s name and frame rate, will give the localization result.')
    parser.add_argument('-i','--input_video',required=True,
                        help='''Name of the input video, including the path and file extension, for example: video/video_test_0000131.mp4''')
    parser.add_argument('-f','--framerate',default=25,type=int,
                        help='''frame rate of the given video, or the number of frames you wish to extract from each second
                                (Default: 25)''')
    # parse input arguments
    args = parser.parse_args()
    input_video = args.input_video
    framerate = args.framerate
    videoname = os.path.splitext(os.path.basename(input_video))[0] # only the video's name
    framedir = 'frame/'
    preddir = 'pred/'

    # make directories and remove the existing result directory
    if os.path.exists(framedir + videoname):
        shutil.rmtree(framedir + videoname)
    if os.path.exists(preddir+'pro/output/'):
        shutil.rmtree(preddir+'pro/output/')
    if os.path.exists(preddir+'loc/output/'):
        shutil.rmtree(preddir+'loc/output/')
    if os.path.exists(preddir+'final/'):
        shutil.rmtree(preddir+'final/')
	
    os.makedirs(framedir + videoname)
    os.makedirs(preddir+'pro/output/')
    os.makedirs(preddir+'loc/output/')
    os.makedirs(preddir+'final/')

    # # -------------------------- # #
    # # --- segment generation --- # #
    # # -------------------------- # #

    start_time = time.time()
    print 'extract frames starts'   
    # extract frames
    cmd = '../lib/preprocess/ffmpeg -i ' + input_video + ' -r ' + str(framerate) + ' -f image2 ' + framedir + videoname +'/%06d.jpg' + ' 2>' + framedir + 'frame_extract.log'
    os.system(cmd)
    num_frame = len(os.listdir(framedir+videoname))
    print 'extract frames done in '+str(time.time() - start_time)[0:6]+' s'

    start_time = time.time()
    print 'init sliding window starts'
	
    # initial seg_swin: see function below
    seg_swin = swin_init(videoname, framerate, num_frame)
    print 'init sliding window done in '+str(time.time() - start_time)[0:6]+' s'
    
    # # ------------------------ # #
    # # --- proposal network --- # #
    # # ------------------------ # #
	
    start_time = time.time()
    print 'generate proposal list starts'
    # generate proposal list
    fout1 = open('pred/pro/demo_list_test_prefix_proposal.lst','w')
    fout2 = open('pred/pro/demo_list_test_uniform16_proposal.lst','w')
    for i in range(len(seg_swin)):
        fout1.write(preddir+'pro/output/'+'{0:06}'.format(i+1)+'\n')
        fout2.write(framedir + videoname + '/ ' + str(seg_swin[i][2])+' 0 '+str(seg_swin[i][1]/16)+'\n')         
    fout1.close()
    fout2.close()
    print 'generate proposal list done in '+str(time.time() - start_time)[0:6]+' s'

    start_time = time.time()
    print 'run proposal network starts'
    # run proposal network
    os.system('chmod +x ./pred/pro/feature_extract.sh')
    os.system('./pred/pro/feature_extract.sh')
    print 'run proposal network done in '+str(time.time() - start_time)[0:6]+' s'

    start_time = time.time()
    print 'read proposal results starts'
    # read proposal results
    for img_index in range(len(seg_swin)):
        prob = read_binary_blob(preddir+'pro/output/'+'{0:06}'.format(img_index+1)+'.prob')
        seg_swin[img_index][9] = prob[1]

    # write seg_swin ----- first time
    myfile = open(preddir+'pro/seg_swin.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(seg_swin)
    myfile.close()

    print 'read proposal results done in '+str(time.time() - start_time)[0:6]+' s'

    # # ---------------------------- # #
    # # --- localization network --- # #
    # # ---------------------------- # #

    start_time = time.time()
    print 'generate localization list starts'
    # update seg_swin: choose segment that contain information, threshold 0.7
    new_seg_swin = []
    for row in seg_swin:
        if row[9] >= 0.7:
            new_seg_swin.append(row)
    seg_swin = new_seg_swin

    # generate localization list
    fout3 = open('pred/loc/demo_list_test_prefix_localization.lst','w') 
    fout4 = open('pred/loc/demo_list_test_uniform16_localization.lst','w')
    for i in range(len(seg_swin)):
        fout3.write(preddir+'loc/output/'+'{0:06}'.format(i+1)+'\n')
        fout4.write(framedir + videoname +'/ '+ str(seg_swin[i][2]) +' 0 '+ str(seg_swin[i][1]/16) +' 0\n')            
    fout3.close()
    fout4.close()
    print 'generate localization list done in '+str(time.time() - start_time)[0:6]+' s'

    start_time = time.time()
    print 'run localization network starts'
    # run localization network
    os.system('chmod +x ./pred/loc/feature_extract.sh')
    os.system('./pred/loc/feature_extract.sh')
    print 'run localization network done in '+str(time.time() - start_time)[0:6]+' s'

    start_time = time.time()
    print 'read localization results starts'
    # read localization results
    for img_index in range(len(seg_swin)):
        prob = read_binary_blob(preddir+'loc/output/'+'{0:06}'.format(img_index+1)+'.prob') # a list of 21 elements
        (a,b) = max((value,index) for index,value in enumerate(prob)) # choose the largest probability and its index 
        seg_swin[img_index][8] = a # value
        seg_swin[img_index][10] = b # index

    # write seg_swin ----- second time
    myfile = open(preddir+'loc/seg_swin.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(seg_swin)
    myfile.close()
    print 'read localization results done in '+str(time.time() - start_time)[0:6]+' s'

    # # ----------------------- # #
    # # --- post-processing --- # #
    # # ----------------------- # #

    start_time = time.time()
    print 'post-processing starts'
    # update seg_swin: choose segment that are not background
    new_seg_swin = []
    for row in seg_swin:
        if row[10] != 0:
            new_seg_swin.append(row)
    seg_swin = new_seg_swin

    # refine score via window length weights
    wfile = open('../experiments/THUMOS14/win_weight/weight.csv', 'rb')
    weight = [[float(i) for i in row] for row in list(csv.reader(wfile))]
    wfile.close()
    for row in seg_swin:
        row[8] = row[8]*weight[int(log(row[1]/16, 2))][row[10]-1]

    # NMS
    overlap_nms = 0.4
    pick_nms = []
    for cls in range(20):
        zipped = [(idx, [seg_swin[idx][4], seg_swin[idx][5], seg_swin[idx][8]]) for idx, row in enumerate(seg_swin) if row[10]-1==cls]
        if len(zipped) >0:
            [inputpick, valuepick] = zip(*zipped)
        else:
            continue
        pick_nms.extend([inputpick[idx] for idx in nms_temporal(valuepick, overlap_nms)])
    new_seg_swin = []
    new_seg_swin = [seg_swin[idx] for idx in pick_nms]
    seg_swin = new_seg_swin

    # # --------------------- # #
    # # --- output result --- # #
    # # --------------------- # #

    # final localization prediction
    seg_swin = sorted(seg_swin, key=lambda x:x[8]) # sort and get index
    res = [[0]*num_frame for _ in range(20)] # 20 classes, n frames
    for row in seg_swin:
        for item in range(row[2]-1, row[3]+1):
            res[row[10]-1][item] = row[8]

    # write seg_swin ----- third time
    myfile = open(preddir+'final/seg_swin.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(seg_swin)
    myfile.close()

    # write result file
    resfile = open(preddir+'final/res.csv', 'wb')
    wr = csv.writer(resfile, quoting=csv.QUOTE_ALL)
    wr.writerows(res)
    resfile.close()
	
    print 'post-processing done in '+str(time.time() - start_time)[0:6]+' s'



def swin_init(videoname, framerate, num_frame):
    # Initial the seg_swin matrix for this video using the numbers below:
    # 1:video_name,by its id number 2:frame_size_type 3:start_frame 4:end_frame 5:start_time 6:end_time 12:win_overlap_rate
    # seg_swin is a matrix with 12 columns and n rows, where n is the number of segments
    win_overlap_rate = 0.75
    seg_swin = []
    linenum = 0
    for window_stride in [16,32,64,128,256,512]:
        win_overlap = int(window_stride*(1-win_overlap_rate))
        start_frame = 1
        end_frame = window_stride
        while end_frame <= num_frame:
            seg_swin.append([0]*12) # a list of zeros
            seg_swin[linenum][0] = int(videoname.split('_')[-1])
            seg_swin[linenum][1] = window_stride
            seg_swin[linenum][2] = start_frame
            seg_swin[linenum][3] = end_frame
            seg_swin[linenum][4] = float(start_frame)/framerate
            seg_swin[linenum][5] = float(end_frame)/framerate
            seg_swin[linenum][11] = 1-win_overlap_rate
            # prepare for next iteration
            linenum = linenum+1
            start_frame = start_frame + win_overlap
            end_frame = end_frame + win_overlap
    return seg_swin

def read_binary_blob(filename):
    f = open(filename, 'rb')
    s = struct.unpack('iiiii', f.read(20)) # the first five are integers
    length = s[0]*s[1]*s[2]*s[3]*s[4]
    data = struct.unpack('f'*length, f.read(4*length))
    f.close()
    return list(data)


def nms_temporal(boxes, overlap):
    pick = []

    if len(boxes)==0:
        return pick
    
    x1 = [b[0] for b in boxes]
    x2 = [b[1] for b in boxes]
    s = [b[-1] for b in boxes]
    union = map(operator.sub, x2, x1) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick


if __name__ == "__main__":
    main()