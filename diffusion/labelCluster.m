function [ featLabel ] = labelCluster( centers, allfeat, N_sample, nclus )
%��allfeat���վ�������centers���б궨����
%   Detailed explanation goes here

distance = zeros(nclus,N_sample);
%%�������������;������ĵ�֮�������ŷ�Ͼ���
for i=1:nclus
    for j=1:N_sample
        distance(i,j) = norm(allfeat(:,j)-centers(:,i));
    end
end

[minval , featLabel] = min(distance);

end

