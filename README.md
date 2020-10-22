# MSP-IL
Matlab implementation of paper "Motion Status Prediction and Iterative Localization for Visual Object Tracking"

If you have any questions or suggestions, you can contact Runqing Zhang(zrq1993@bupt.edu.cn)

![image](https://github.com/painshine/myRLtrack/tree/master/main_framework.jpg)


Dependencies:
MatConvNet, PDollar Toolbox. 
Please download the latest MatConvNet (http://www.vlfeat.org/matconvnet/) 

RLtrack:
Demo and code for Motion Status Prediction model.
--Demo & Experiments: RLpre_demo.m in https://github.com/painshine/myRLtrack/tree/master/RLtrack/code/myRLtrack
--Formula: RLS_equations in https://github.com/painshine/myRLtrack/tree/master/RLtrack/code
--The model in this version predict 11 kinds of motion state based on reinforcement learning, and we'll update the latest organized code in the near future.

ICF_code:
Demo and code for iterative correlation filters.
--Demo: demo.m in https://github.com/painshine/myRLtrack/tree/master/ICF_code/AutoTrack3
--The code in this project is based on the AutoTrack [CVPR 2020] (https://arxiv.org/pdf/2003.12949.pdf)
We'll update the latest organized code based on the GFS-DCF [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Joint_Group_Feature_Selection_and_Discriminative_Filter_Learning_for_Robust_ICCV_2019_paper.pdf) in the near future.
--Results data: results_base_Autotrack.rar in https://github.com/painshine/myRLtrack/tree/master/ICF_code/AutoTrack3

