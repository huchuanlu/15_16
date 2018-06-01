function [ featLabel ] = labelCluster( centers, allfeat, N_sample, nclus )
%将allfeat按照聚类中心centers进行标定分类
%   Detailed explanation goes here

distance = zeros(nclus,N_sample);
%%计算所有样本和聚类中心点之间的所有欧氏距离
for i=1:nclus
    for j=1:N_sample
        distance(i,j) = norm(allfeat(:,j)-centers(:,i));
    end
end

[minval , featLabel] = min(distance);

end

