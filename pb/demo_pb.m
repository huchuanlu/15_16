%clc;
%clear;
%inpath = 'F:\�о���\2013�����������д\����ʦ�㷢�Ĳ���\Image1000\*.jpg';
%matpath = '.\mat\';
%Modify begin
function pb_bi=demo_pb(image)
outpath = '.\bi_result\';
%dir_im = dir(inpath);
%for i =16:length(dir_im)
    %image_name = dir_im(i).name;
    I = im2double(image);
    %I = im2double(I);
    
    [pb,theta] = pbCGTG(I);
    level=graythresh(pb);
    pb_bi=(pb>level);
    %save([matpath,image_name(1:end-4),'_pb.mat'],'pb');
   % imwrite(pb_bi,[outpath,image_name(1:end-4),'_pb.png'],'png');
%end
%Modify end

