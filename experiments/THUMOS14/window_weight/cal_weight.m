% ----------------------------------------------------------------------------------------------------------------
% Segment-CNN
% Copyright (c) 2016 , Digital Video & Multimedia (DVMM) Laboratory at Columbia University in the City of New York.
% Licensed under The MIT License [see LICENSE for details]
% Written by Zheng Shou, Dongang Wang, and Shih-Fu Chang.
% ----------------------------------------------------------------------------------------------------------------

load class.mat;
filepath = '../annotation/annotation_val/';
window_length = [0.6, 1.24, 2.52, 5.08, 10.20, 22.44];
window_cnt = zeros(6,20);

for i = 1:20
    f = fopen([filepath, classid{i}, '_val.txt']);
    while ~feof(f)
        line = fgetl(f);
        line = regexp(line,' ','split');
        delta = str2double(line{4})-str2double(line{3});
        for j = 1:6
            if delta>=(window_length(j)/2) && delta <= (window_length(j)*2)
                window_cnt(j,i) = window_cnt(j,i)+1;
            end
        end
    end
end

weight = window_cnt./repmat(sum(window_cnt,1),6,1);
save('weight.mat','weight');
save('count.mat','window_cnt');



