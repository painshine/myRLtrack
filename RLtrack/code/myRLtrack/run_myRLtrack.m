function result = run_myRLtrack(seq,rp,imagebsave)
    addpath('test/');
    addpath(genpath('utils/'));

    init_settings;

    run(matconvnet_path);

    %load('models/net_rl.mat');
    load('models/net.mat');

    opts.visualize = true;
    opts.printscreen = true;

    rng(1004);
    opts.finetune_interval = 70;
    
    vid_path = seq.path;
    
    [results, t, p] = myRLtrack(net, vid_path, opts);
    
    result.res = results;
    result.type = 'rect';
    result.fps = t;

end