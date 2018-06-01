clc;
clear all;
addpath('k_means');
addpath('pb');
addpath('diffusion');

seg_paras = [
         	 200  20;
           	 250  25;
           	 300  25;
             350  30;
             400  30;
           	];    
isLAB = 1;                              
isRGB = 0;
isXY  = 0;
featDim = 3*(isLAB+isRGB)+2*isXY;       

paramPropagate.lamna = 0.5;            
paramPropagate.nclus = 8;
paramPropagate.maxIter=500;            
 

inpath = '.\Image\*.bmp';
superpixel_path = '.\FinalResult\superpixel\';
pb_path = '.\FinalResult\bi_result\*.png'; 
background_seeds_map_path='.\FinalResult\background_seeds_map\';
background_based_map_path = '.\FinalResult\background_based_map\';
gauss_map_path='.\FinalResult\gauss_map\';
foreground_based_map_path = '.\FinalResult\foreground_based_map\';
unified_map_path='.\FinalResult\unified_map\';
refined_map_path='.\FinalResult\refined_map\';

 mkdir(pb_path(1:end-5));
 mkdir(background_seeds_map_path);

 mkdir(background_based_map_path);
 mkdir(gauss_map_path);

 mkdir(foreground_based_map_path);
 mkdir(unified_map_path);
 mkdir(superpixel_path);
  mkdir(refined_map_path);
 
dir_im = dir(inpath);
lamda =1;
for i = 1:length(dir_im)
    imName = dir_im(i).name;
    input_im=im2double(imread([inpath(1:end - 5) imName]));
 %% obtain border
      pb_im=demo_pb(input_im);
      imwrite(pb_im,[pb_path(1:end-5),imName(1:end-4),'_pb.png'],'png');
 %% SLIC
     spnumber = seg_paras(1,1);
      compactness = seg_paras(1,2);
      comm=['SLICSuperpixelSegmentation' ' ' [inpath(1:end - 5) imName] ' ' int2str(compactness) ' ' int2str(spnumber) ' ' superpixel_path];
       system(comm); 

%% background seeds
    [row,col,dimen] = size(input_im); 
    row_thes = 0.1*row;
    col_thes = 0.1*col;
    superlabel = ReadDAT(size(input_im),[superpixel_path imName(1:end - 4) '.dat']);
    STATS = regionprops(superlabel, 'all');
    sup_num = numel(STATS);
    pb_weight = zeros(sup_num,1);
    input_imlab = rgb2lab(input_im);
    L = input_imlab(:,:,1);
    A = input_imlab(:,:,2);
    B = input_imlab(:,:,3);
        R = input_im(:,:,1);
    G = input_im(:,:,2);
    B1 = input_im(:,:,3);
    
    image = [];
    mat_temp = [];
    for j = 1:sup_num
        mask = superlabel == j;
        boundary = edge(mask).*pb_im;
        idx = find(edge(mask) == 1);
        weight = sum(sum(boundary))/length(idx);
        pb_weight(j) = weight;
        pixelind = STATS(j).PixelIdxList;
        indxy = STATS(j).PixelList; 
        pos_mat(j,:) = [mean(indxy(:,1)),mean(indxy(:,2))];
        color_mat(j,:) = [mean(L(pixelind)),mean(A(pixelind)),mean(B(pixelind))];
    end
    thresh = graythresh(pb_weight);

    boundary_index = [];
    guided_input = ones(row,col);
     background_seeds_map=zeros(row,col,3);
    
    for j = 1: sup_num
        center = STATS(j).Centroid;
        if (center(2)<row_thes||center(2)>(row-row_thes)||center(1)<col_thes||center(1)>(col-col_thes))
            if pb_weight(j)<thresh
            boundary_index = [boundary_index j];
            pixelind = STATS(j).PixelIdxList;
            guided_input(pixelind) = 0;
            end
        end
    end
   
   background_seeds_map(:,:,1)=R.*guided_input;
  background_seeds_map(:,:,2)=G.*guided_input;
   background_seeds_map(:,:,3)=B1.*guided_input;
   
  imwrite(background_seeds_map,[background_seeds_map_path,imName(1:end-4),'.jpg'],'jpg');
%% background-based saliency detection
  pos_mat(:,1) = pos_mat(:,1)/col;
    pos_mat(:,2) = pos_mat(:,2)/row;

 vector_temp=[];
    for j = 1:numel(boundary_index)
        harris_sp_label = boundary_index(j);
         harris_sp_color =color_mat(harris_sp_label,:);
        harris_sp_pos = pos_mat(harris_sp_label,:);    
        for q = 1:sup_num
            theta = 1;
            cur_sp_color = color_mat(q,:);
            cur_sp_pos = pos_mat(q,:);
            if(harris_sp_label == q)
                sal_temp = 0;
            else
            d_color = sqrt(sum((harris_sp_color - cur_sp_color).^2 ));
            d_space = sqrt(sum((harris_sp_pos - cur_sp_pos).^2));
             
             sal_temp = d_color*(1-d_space);
            end
            vector_temp(q) = sal_temp;
        end
        mat_temp(:,j) = vector_temp;
    end
for n = 1:numel(boundary_index)
    harris_sp_label = boundary_index(n);
    mat_temp(harris_sp_label,n) = sum(mat_temp(harris_sp_label,:))/(numel(boundary_index)-1);
end
sal_vector = mean(mat_temp,2);
%
salGau1_vector=zeros(sup_num,1);
sal = zeros(row,col);
salGau1 = zeros(row,col);
for  j = 1:sup_num
    pixelind = STATS(j).PixelIdxList;
    sal(pixelind) = sal_vector(j); 
end
sal_vector = (sal_vector - min(sal_vector(:)))/(max(sal_vector(:)) - min(sal_vector(:)));
sal = (sal - min(sal(:)))/(max(sal(:)) - min(sal(:)));
%
imwrite(sal, [background_based_map_path imName(1:end-4),'.jpg'], 'jpg');
%% Gauss
 thresh = graythresh(sal_vector);
 se_sp = sal_vector >thresh;
 se_sp_idx = find(se_sp == 1);

   gauss_map = zeros(row,col);
  for xx = 1:row
    for yy = 1:col
        gauss_map(xx,yy) = exp(-9*((xx/row-0.5)^2+(yy/col-0.5)^2)); 
    end
end

 gauss_map = (gauss_map - min(gauss_map(:)))/(max(gauss_map(:)) - min(gauss_map(:)));
 imwrite(gauss_map, [gauss_map_path imName(1:end-4),'.jpg'], 'jpg');
%%  foreground-based saliency detection
    r=size(input_im,1);
    c=size(input_im,2);
     regions = calculateRegionProps(sup_num,superlabel);
  [sup_feat color_weight] = extractSupfeat(input_im,input_imlab,regions,r,c,sup_num); 
    feat = chooseFeature(sup_feat,isRGB,isLAB,isXY);

 
 color_mat = (color_mat - min(color_mat(:)))/(max(color_mat(:)) - min(color_mat(:)));
 mat_temp = [];
 for j = 1:numel(se_sp_idx)
    cur_sp_label = se_sp_idx(j);
     cur_sp_color = color_mat(cur_sp_label,:);
     cur_sp_pos = pos_mat(cur_sp_label,:);
     for q = 1:sup_num
         se_sp_color = color_mat(q,:);
         se_sp_pos = pos_mat(q,:);
         if (cur_sp_label == q)
             sal_temp = 0;
         else
             d_color = sqrt(sum((cur_sp_color - se_sp_color).^2 ));
             d_space = sqrt(sum((cur_sp_pos - se_sp_pos).^2));
             sal_temp =1/(d_color+d_space);
         end
         vector_temp(q) = sal_temp;
    end
     mat_temp(:,j) = vector_temp;
 end
 for n = 1:numel(se_sp_idx)
     cur_sp_label = se_sp_idx(n);
     mat_temp(cur_sp_label,n) = mean(mat_temp(cur_sp_label,:));
 end
 %
 sal_vector = mean(mat_temp,2);
 sal = zeros(row,col);
 salGau2 = zeros(row,col);
 for  j = 1:sup_num
     pixelind = STATS(j).PixelIdxList;
     sal(pixelind) = sal_vector(j); 
    
 end
 sal = (sal - min(sal(:)))/(max(sal(:)) - min(sal(:)));
 %
 imwrite(sal, [foreground_based_map_path imName(1:end-4),'.jpg'], 'jpg');

 %% unification
 sal_1=im2double(imread([background_based_map_path imName(1:end-4) ,'.jpg']));
    sal_2=im2double(imread([foreground_based_map_path imName(1:end-4), '.jpg']));
      smap=sal_1.*(1-exp(-6*(sal_2).^1));
   smap = (smap - min(smap(:))) / (max(smap(:)) - min(smap(:)));
     imwrite(smap,[unified_map_path,imName(1:end-4), '.jpg'],'jpg');
     
  %% Refinement
  Unification_map=im2double(imread([unified_map_path imName(1:end-4),'.jpg']));
   Gauss_map=im2double(imread([gauss_map_path imName(1:end-4),'.jpg']));
 
   Unification_map_error=zeros(sup_num,1);
   for  j = 1:sup_num
     pixelind = STATS(j).PixelIdxList;
     Unification_map_error(j) = mean(mean(Unification_map(pixelind))); 
   end
   propUnification_map_error = descendPropagation(feat,Unification_map_error,paramPropagate,sup_num,featDim); 
    refined_map=zeros(r,c);
 for  j = 1:sup_num
     pixelind = STATS(j).PixelIdxList;
     refined_map(pixelind) = propUnification_map_error(j); 
 end
 
 refined_map=refined_map.*Gauss_map;
 imwrite(refined_map, [refined_map_path , imName(1:end-4),'.jpg'], 'jpg');
   i
end
   
    
    
    
    