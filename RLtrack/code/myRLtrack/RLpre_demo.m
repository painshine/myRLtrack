
addpath('test/');
addpath(genpath('utils/'));
vid_name = 'Freeman1';
vid_path = 'data/Freeman1';

init_settings;

run(matconvnet_path);

%load('models/net_rl.mat');
load('models/net.mat');

opts.visualize = true;
opts.printscreen = true;

rng(1004);

gt_path = fullfile(vid_path,'groundtruth_rect.txt');
gt = importdata(gt_path);

[seq_len,~] = size(gt);
old_err_list = [];
my_err_list = [];

for num_frame = 1:seq_len
    if num_frame == 1
        old_err = 0;
        my_err = 0;
    else
        input_box_old = gt(num_frame-1,:);
        %% generate history action labels
        % init history state
        action_history = zeros(opts.num_show_actions, 1);
        action_history_oh_zeros = zeros(opts.num_actions*opts.num_action_history, 1);
        action_history_oh = action_history_oh_zeros;

        % generate input history state
        if num_frame < 12
            action_history = zeros(opts.num_show_actions, 1);
            action_history = zeros(opts.num_show_actions, 1);
            action_history_oh_zeros = zeros(opts.num_actions*opts.num_action_history, 1);
            action_history_oh = action_history_oh_zeros;
        else
           for action_frame = num_frame-10: num_frame-1
               old_bbox = gt(action_frame-1,:);
               current_bbox = gt(action_frame,:);
               tmp_action11bit = gen_action_labels(opts.num_actions, opts, current_bbox, old_bbox);
               [~,tmp_action1bit] = max(tmp_action11bit(1:end));

               % update history state
               action_history(2:end) = action_history(1:end-1);
               action_history(1) = tmp_action1bit;
           end
        end

        %% demo prediction search region
        predict_bbox = RLpre(net, vid_path, opts,num_frame,input_box_old,action_history);

        %% comparision
        input_box_old;
        predict_bbox;
        gt_res = gt(num_frame,:);

        old_err = cal_center_err(input_box_old,gt_res);
        my_err = cal_center_err(predict_bbox,gt_res);
    end
    old_err_list = [old_err_list;old_err];
    my_err_list = [my_err_list;my_err];
end

figure();
grid on;
plot(old_err_list);
hold on;
plot(my_err_list);

avg_old = mean(old_err_list);
avg_new = mean(my_err_list);
legend1 = ['baseline ' '(' num2str(avg_old) ')'];
legend2 = ['RL\_sequential\_prediction ' '(' num2str(avg_new) ')'];
legend(legend1,legend2);
title_name = ['Search Region Center Pixel Errors for ' vid_name];
title(title_name);
