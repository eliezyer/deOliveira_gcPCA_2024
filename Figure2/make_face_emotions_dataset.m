clear
close all
cd('C:\Users\fermi\Dropbox\preprocessing_data\gcPCA_files\face\CFD_V3\Images\CFD')
aux = dir;
isfolder=[aux(:).isdir];
aux_filt = aux(isfolder);
aux_filt = aux_filt(3:end);
%picking only sessions with the figures
ses2use = [];
for ses = 1:length(aux_filt)
    aux_2 = dir(aux_filt(ses).name);
    temp_hc = cellfun(@(x) ~isempty(strfind(x,'-HC')),{aux_2(:).name});
    temp_angry = cellfun(@(x) ~isempty(strfind(x,'-A')),{aux_2(:).name});
    temp_male = cellfun(@(x) ~isempty(strfind(x,'M-')),{aux_2(:).name});
    if sum(temp_hc)>0 && sum(temp_angry)>0 && sum(temp_male)>0
        ses2use = [ses2use,ses];
    end
end
aux_final = aux_filt(ses2use);
neutral_faces = [];
ho_faces = [];
hc_faces = [];
angry_faces = [];
fear_faces = [];
angry_files = [];
hc_files = [];
for ses = 1:length(aux_final)
    aux_2 = dir(aux_final(ses).name);
    %loading neutral and saving
    temp_ans = cellfun(@(x) ~isempty(strfind(x,'-N.')),{aux_2(:).name});
    if sum(temp_ans)>0
        temp_img = imread(fullfile(aux_final(ses).name,aux_2(find(temp_ans)).name));
        temp_img = temp_img(175:end,500:2000,:);
        grey_img = imresize(rgb2gray(temp_img),[246,186]);
        neutral_faces = cat(3,neutral_faces,grey_img);
    end
    %loading happy open and saving
    temp_ans = cellfun(@(x) ~isempty(strfind(x,'-HO.')),{aux_2(:).name});
    if sum(temp_ans)>0
        temp_img = imread(fullfile(aux_final(ses).name,aux_2(find(temp_ans)).name));
        temp_img = temp_img(175:end,500:2000,:);
        grey_img = imresize(rgb2gray(temp_img),[246,186]);
        ho_faces = cat(3,ho_faces,grey_img);
    end
    %loading happy closed and saving
    temp_ans = cellfun(@(x) ~isempty(strfind(x,'-HC.')),{aux_2(:).name});
    if sum(temp_ans)>0
        temp_img = imread(fullfile(aux_final(ses).name,aux_2(find(temp_ans)).name));
        temp_img = temp_img(175:end,500:2000,:);
        grey_img = imresize(rgb2gray(temp_img),[246,186]);
        hc_faces = cat(3,hc_faces,grey_img);
        hc_files = [hc_files; fullfile(aux_final(ses).name,aux_2(find(temp_ans)).name)];
    end
    %loading fear and saving
    temp_ans = cellfun(@(x) ~isempty(strfind(x,'-F.')),{aux_2(:).name});
    if sum(temp_ans)>0
        temp_img = imread(fullfile(aux_final(ses).name,aux_2(find(temp_ans)).name));
        temp_img = temp_img(175:end,500:2000,:);
        grey_img = imresize(rgb2gray(temp_img),[246,186]);
        fear_faces = cat(3,fear_faces,grey_img);
    end
    %loading angry and saving
    temp_ans = cellfun(@(x) ~isempty(strfind(x,'-A.')),{aux_2(:).name});
    if sum(temp_ans)>0
        angry_files = [angry_files; fullfile(aux_final(ses).name,aux_2(find(temp_ans)).name)];
        temp_img = imread(fullfile(aux_final(ses).name,aux_2(find(temp_ans)).name));
        temp_img = temp_img(175:end,500:2000,:);
        grey_img = imresize(rgb2gray(temp_img),[246,186]);
        angry_faces = cat(3,angry_faces,grey_img);
    end
end
%% making ellipse mask to cut the figures;
X0=0; %Coordinate X
Y0=-3; %Coordinate Y
l=75; %Length
w=45; %Width
phi=90; %Degree you want to rotate
[X,Y] = meshgrid(-122:123,-92:93); %make a meshgrid: use the size of your image instead
ellipse = ((X-X0)/l).^2+((Y-Y0)/w).^2<=1; %Your Binary Mask which you multiply to your image, but make sure you change the size of your mesh-grid
RotateEllipse = imrotate(ellipse,phi);
% figure;imagesc(RotateEllipse)
%% run PCA on all faces together
labels = cat(1,zeros(size(hc_faces,3),1),...
    ones(size(angry_faces,3),1));
data_A = double(cat(3,hc_faces,angry_faces));
data_A = data_A .* repmat(RotateEllipse,[1,1,size(hc_faces,3)+size(angry_faces,3)]);
data_B = double(neutral_faces);
data_B = data_B .* repmat(RotateEllipse,[1,1,size(neutral_faces,3)]);
m.data_A      = data_A;
m.data_B      = data_B;
m.EllipseMask = RotateEllipse;
m.data_info   = 'images in NxMxP, where NxM are pixels and P are images indices, data_A is with emotional expression and data_B are neutral faces';
m.labels      = labels;
m.labels_info = 'labels of each image, matches P in data, zeros is happy closed faces and ones are angry faces';
face_emotions = m;
save('face_emotions.mat','face_emotions')