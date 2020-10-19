function train_RLtrack()
% ADNET_TRAIN Train the ADNet 
%
% Sangdoo Yun, 2017.

addpath('train/');
addpath(genpath('utils/'));
init_settings;
run(matconvnet_path);

rng(1004);

% Training stage 1: SL
cd('./models');
if exist('net_sl.mat')
    load net_sl.mat
    opts.vgg_m_path = vgg_m_path;
    train_videos = get_train_videos(opts);
else
    opts.vgg_m_path = vgg_m_path;
    [net, train_videos] = adnet_train_SL(opts);
    save('./models/net_sl.mat', 'net')
end
cd('../');

% Training stage 2: RL
net = adnet_train_RL(net, train_videos, opts);
save('./models/net.mat', 'net')


