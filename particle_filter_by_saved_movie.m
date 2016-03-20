%% Parameters
% reference : http://www.mathworks.com/matlabcentral/fileexchange/33666-simple-particle-filter-demo
clear all;
close all;
clc;

F_update = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];

Npop_particles = 4000;  

Xstd_rgb = 50; %����
Xstd_pos = 25;
Xstd_vec = 5;

Xrgb_trgt = [255; 0; 0];  %������������˲���ɫ��rgb��Ϣ��׷�ٵġ�

%% Loading Movie

vr = VideoReader('Person.wmv');

Npix_resolution = [vr.Width vr.Height];
Frame_velocity = vr.FrameRate;
Nfrm_movie = floor(vr.Duration * vr.FrameRate);  %֡��������floorΪ�������롣

%% Object Tracking by Particle Filter

X = create_particles(Npix_resolution, Npop_particles); % ���ӳ�ʼ�����ڻ����в������ȷֲ����������

for k = 1:Nfrm_movie
    
    % Getting Image
    Y_k = read(vr, k);  %ע��˴���Y_kΪһ����ά���飬�ֱ��ʺͿ��ܵ�3ͨ����
    imshow(Y_k);
    
   % show_particles(X, Y_k);
    
    % Forecasting
    %ͨ��״̬ģ��Ԥ��  ������õ�������һʱ�̻����ϵ�������
    X = update_particles(F_update, Xstd_pos, Xstd_vec, X); 
    
  %  show_particles(X, Y_k);
    
    % Calculating Log Likelihood
    L = calc_log_likelihood(Xstd_rgb, Xrgb_trgt, X(1:2, :), Y_k);
    
  %  show_particles(X, Y_k);
    
    % Resampling
    X = resample_particles(X, L);

    % Showing Image
    show_particles(X, Y_k); 
%    show_state_estimated(X, Y_k);

end

