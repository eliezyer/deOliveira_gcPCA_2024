% script to get the session from Girardeau dataset to test cPCA/gcPCA
clear;close all;clc
session_folder = '/mnt/probox/buzsakilab.nyumc.org/datasets/GirardeauG/Rat11/Rat11-20150328';
save_folder = '/home/eliezyer/Documents/github/deOliveira_gcPCA_2024/Figure3/';

% create a structure with all the necessary info

%% loading necessary data
[~,basename] = fileparts(session_folder);
if exist(fullfile(session_folder,[basename '.pos.mat']),'file')
    load(fullfile(session_folder,[basename '.pos.mat']))
    load(fullfile(session_folder,[basename  '.spikes.cellinfo.mat']))
    load(fullfile(session_folder,[basename  '.task.states.mat']))
    
    temp_airpuff = readtable(fullfile(session_folder,[basename  '.puf.evt']),'FileType','text');
    if isfield(spikes,'region')
        temp_pr_idx = cellfun(@(x) ~isempty(strfind(x,'-prerun')),task.states.labels);
        temp_r_idx  = cellfun(@(x) ~isempty(strfind(x,'-run')),task.states.labels);
        temp_psr_idx  = cellfun(@(x) ~isempty(strfind(x,'-postrun')),task.states.labels);
        temp_pre_task_idx  = cellfun(@(x) ~isempty(strfind(x,'-presleep')),task.states.labels);
        temp_post_task_idx  = cellfun(@(x) ~isempty(strfind(x,'-postsleep')),task.states.labels);
        if sum(temp_pr_idx)>0 && sum(temp_r_idx)>0 && sum(temp_psr_idx)>0 && sum(temp_pre_task_idx)>0 && sum(temp_post_task_idx)>0
            %% getting periods of task
            pre_run_ints = task.states.ints(ceil(find(temp_pr_idx,1)/2),:);
            run_ints     = task.states.ints(ceil(find(temp_r_idx,1)/2),:);
            post_run_ints     = task.states.ints(ceil(find(temp_psr_idx,1)/2),:);
            pre_task_ints     = task.states.ints(ceil(find(temp_pre_task_idx,1)/2),:);
            post_task_ints     = task.states.ints(ceil(find(temp_post_task_idx,1)/2),:);
            
            %% if spike region is empty than input with 'none'
            spikes_region = spikes.region;
            idx2change = cellfun(@(x) isempty(x),spikes_region);
            spikes_region(idx2change) = {'nolabel'};
            %% preparing output structure
            m.spikes_ts          = spikes.times;
            m.spikes_ts_info     = 'spike times in s';
            m.spikes_region      = spikes_region;
            m.spikes_region_info = 'brain region where the spikes comes from';
            m.location           = cat(2,pos.X.data,pos.Y.data);
            m.location_info      = 'All two (x,y) axes of tracking location of the animal';
            m.linspd             = pos.linSpd.data;
            m.linspd_info        = 'Animal running speed based on x, y coordinates';
            m.tracking_t         = pos.linSpd.t;
            m.tracking_t_info    = 'Timestamps of the tracking, good to be used with the speed and the location';
            
            m.task.run_intervals = run_ints;
            m.task.pre_task_intervals = pre_task_ints;
            m.task.post_task_intervals = post_task_ints;
            m.task.states = task.states;
            
            m.air_puff_times = temp_airpuff.Var1;
            m.session_folder_name= session_folder;
        end
    end
end

%% saving it
hpc_gg_session = m;
cd(save_folder)
save('hpc_gg_session.mat','hpc_gg_session')