function [  ] = run_demo( videoname, framerate )
% Demo of Segment-CNN to predict localization results for a given video.

% ----------------------------------------------------------------------------------------------------------------
% Segment-CNN
% Copyright (c) 2016 , Digital Video & Multimedia (DVMM) Laboratory at Columbia University in the City of New York.
% Licensed under The MIT License [see LICENSE for details]
% Written by Zheng Shou, Dongang Wang, and Shih-Fu Chang.
% ----------------------------------------------------------------------------------------------------------------

% Example: run_demo('video/video_test_0000131.mp4',25)

%% check and set input. some init.
if nargin > 2
    error('max number of input param: 2');
end
switch nargin
    case 0
        videoname = 'video_test_0000131';
        videotype = 'mp4';
        framerate = 25;
    case 1
	    tmp1 = strsplit(videoname,'.');
		tmp2 = strsplit(tmp1{1},'/');
	    videoname = tmp2{2};
	    videotype = tmp1{2};
	    framerate = 25;
    case 2
	    tmp1 = strsplit(videoname,'.');
		tmp2 = strsplit(tmp1{1},'/');
	    videoname = tmp2{2};
	    videotype = tmp1{2};
end

videodir = 'video/';
framedir = 'frame/';
preddir = 'pred/';

addpath(genpath('../lib/'));

if exist([framedir videoname])
    system(['rm -R ' framedir videoname]);
end
system(['mkdir ' framedir videoname]);
if exist([preddir 'pro/output'])
    system(['rm -R ' preddir 'pro/output']);
end
system(['mkdir ' preddir 'pro/output']);
if exist([preddir 'loc/output'])
    system(['rm -R ' preddir 'loc/output']);
end
system(['mkdir ' preddir 'loc/output']);
if exist([preddir 'final'])
    system(['rm -R ' preddir 'final']);
end
system(['mkdir ' preddir 'final']);

%% frame extract
fprintf(['frame extract starts']);
tic;
cmd = ['../lib/preprocess/ffmpeg -i ' videodir videoname '.' videotype ' -r ' num2str(framerate) ...
	' -f image2 ' framedir videoname '/' '%06d.jpg 2>' framedir 'frame_extract.log'];
system(cmd);

fprintf(['frame extract done in ' num2str(toc) ' s\n']);

%% init sliding window
% 1:video_name 2:frame_size_type 3:start_frame 4:end_frame 5:start_time 6:end_time 12:win_overlap_rate
fprintf(['init sliding window starts']);
tic;
seg_swin = zeros(0,12);
win_overlap_rate = 0.75;
img = dir([ framedir videoname '/*.jpg']);
for window_stride=[16,32,64,128,256,512]
    win_overlap = window_stride*(1-win_overlap_rate);
    start_frame = 1;
    end_frame = window_stride;
    while end_frame <= length(img)
        tmp = strsplit(videoname,'_');
        seg_swin(end+1,1) = str2num(tmp{end});
        seg_swin(end,2) = window_stride;
        seg_swin(end,3) = start_frame;
        seg_swin(end,4) = end_frame;
        seg_swin(end,5) = start_frame/framerate;
        seg_swin(end,6) = end_frame/framerate;
        seg_swin(end,12) = 1-win_overlap_rate;                                                                                                                                                                                        
        % next
        start_frame = start_frame + win_overlap;
        end_frame = end_frame + win_overlap;
    end
end

fprintf(['init sliding window done in ' num2str(toc) ' s\n']);

%% generate proposal list
fprintf(['generate proposal starts']);
tic;
fout1 = fopen('pred/pro/demo_list_test_prefix_proposal.lst','w'); 
fout2 = fopen('pred/pro/demo_list_test_uniform16_proposal.lst','w');
for i=1:length(seg_swin)
    fprintf(fout1,['pred/pro/output/' num2str(i,'%06d') '\n']);
    fprintf(fout2,[framedir videoname '/ ' num2str(seg_swin(i,3)) ' 0 ' num2str(seg_swin(i,2)/16) '\n']);            
end
fclose(fout1);
fclose(fout2);

fprintf(['generate proposal list done in ' num2str(toc) ' s\n']);

%% run proposal network
fprintf(['run proposal network starts\n']);
tic;

system('chmod +x ./pred/pro/feature_extract.sh');
system('./pred/pro/feature_extract.sh');

fprintf(['run proposal network done in ' num2str(toc) ' s\n']);

%% read proposal results
fprintf(['read proposal results starts']);
tic;
prob = zeros(size(seg_swin,1),2);
img = dir( [preddir 'pro/output/'] ); % be careful whether all are jpg
for img_index = 3:size(img,1)
    [~,prob(img_index-2,:)] = read_binary_blob([preddir 'pro/output/' img(img_index).name]);
end
seg_swin(:,10) = prob(:,2);
save('pred/pro/seg_swin.mat','seg_swin','-v7.3');

fprintf(['read proposal results done in ' num2str(toc) ' s\n']);

%% generate localization list
fprintf(['generate localization list starts\n']);
tic;
seg_swin = seg_swin(seg_swin(:,10)>=0.7,:);

fout3 = fopen('pred/loc/demo_list_test_prefix_localization.lst','w'); 
fout4 = fopen('pred/loc/demo_list_test_uniform16_localization.lst','w');
for i=1:length(seg_swin)
    fprintf(fout3,['pred/loc/output/' num2str(i,'%06d') '\n']);
    fprintf(fout4,[framedir videoname '/ ' num2str(seg_swin(i,3)) ' 0 ' num2str(seg_swin(i,2)/16) ' 0\n']);            
end
fclose(fout3);
fclose(fout4);

fprintf(['generate localization list done in ' num2str(toc) ' s\n']);

%% run localization results
fprintf(['run localization results starts\n']);
tic;

system('chmod +x ./pred/loc/feature_extract.sh');
system('./pred/loc/feature_extract.sh');

fprintf(['run localization results done in ' num2str(toc) ' s\n']);

%% read localization results
fprintf(['read localization results starts\n']);
tic;
prob = zeros(size(seg_swin,1),21);
img = dir( [preddir 'loc/output/'] ); % be careful whether all are jpg
for img_index = 3:size(img,1)
    [~,prob(img_index-2,:)] = read_binary_blob([preddir 'loc/output/' img(img_index).name]);
end
[a,b] = max(prob(:,:)');
seg_swin(:,9) = a;
seg_swin(:,11) = b-1;
save('pred/loc/seg_swin.mat','seg_swin','-v7.3');

fprintf(['read localization results done in ' num2str(toc) ' s\n']);

%% post-processing
fprintf(['post-processing starts\n']);
tic;
seg_swin=seg_swin(seg_swin(:,11)~=0,:);

% refine score via window length weights
load('../experiments/THUMOS14/win_weight/weight.mat');
for i=1:length(seg_swin)
    seg_swin(i,9)=seg_swin(i,9).*weight(log2(seg_swin(i,2)/16)+1,seg_swin(i,11));
end

% NMS
overlap_nms = 0.4;
pick_nms = [];
for cls=1:20
    inputpick = find((seg_swin(:,11)==cls));
    pick_nms = [pick_nms; inputpick(nms_temporal([seg_swin(inputpick,5) ...
        ,seg_swin(inputpick,6),seg_swin(inputpick,9)],overlap_nms))]; 
end
seg_swin = seg_swin(pick_nms,:);

% rank score by overlap score
[~,order] = sort(-seg_swin(:,9));
seg_swin = seg_swin(order,:);

% merge into final tagging results
res = zeros(20,length(dir([framedir videoname '/']))-2);
for i=size(seg_swin,1):-1:1
    res(seg_swin(i,11),seg_swin(i,3):seg_swin(i,4)) = seg_swin(i,9);
end
save('pred/final/seg_swin.mat','seg_swin','res','-v7.3');

fprintf(['post-processing done in ' num2str(toc) ' s\n']);

end

