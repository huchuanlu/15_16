function propagatedData = descendPropagation(feat,initData,paramPropagate,Nsample,featDim)

%% �����������
centers = form_codebook(feat', paramPropagate.nclus,paramPropagate.maxIter);
%% ���ݾ�������Ϊÿ�������궨�����
[ featLabel ] = labelCluster( centers, feat', Nsample, paramPropagate.nclus );

%% ����sigma^2����Ϊsigma2
meanfeat = mean(feat,1);
sig2 = zeros(1,featDim);
for k=1:featDim
    sig2(k) = norm(feat(:,k) - meanfeat(:,k))^2/Nsample;
end
sigma2 = mean(sig2);

%% ����ÿ����������֮���ŷ�Ͼ���
distMatrix = zeros(Nsample, Nsample);
for i=1:Nsample
    for j=i+1:Nsample
        distMatrix(i,j) = exp(-norm(feat(i,:)-feat(j,:))^2/(2*sigma2));      %������exp_��
        distMatrix(j,i) = distMatrix(i,j);
    end
end

%% �������saliencyֵ���н������У�Ȼ����˳����㴫�����saliencyֵ�����߸���
[desData desInd] = sort(initData);
for i=Nsample:-1:1
    dataLabel = desInd(i);
    clusterlabel = featLabel(dataLabel);
    clusterbgsup = find(featLabel==clusterlabel);
    nInnerCluster = length(clusterbgsup);

	sumdist = 0;
	sumA = 0;
	for m=1:nInnerCluster
        M = clusterbgsup(m);
        sumdist = sumdist + distMatrix(dataLabel,M)*initData(M);
        sumA = sumA + distMatrix(dataLabel,M);
	end
    
    if sumA==0
        sumA = sumA+eps;
    end
	initData(dataLabel)=(1-paramPropagate.lamna)*initData(dataLabel) + paramPropagate.lamna/sumA*sumdist;
end
propagatedData = initData;
propagatedData = (propagatedData - min(propagatedData(:)))/(max(propagatedData(:)) - min(propagatedData(:)));