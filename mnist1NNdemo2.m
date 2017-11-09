
%% Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
%% This file is modified from pmtk3.googlecode.com

load('mnistData');
% set training & testing 
q=1;
samples = 1:1000;
n = length(samples);
error=zeros(5,1);
T=[3,10,50,100,1000];
for m =1:5
    s = 0;
    clear fold;
    clear tranndx;
    x = T(m);
    if(x==3)
    n=999;
    else
        n =1000;
    end
for j = 1:x
        fold(j,:) = randi(1000,[1  (n/x)]);
end
for p = 1:x
    q=1;
    for k = 1:x
        if (k == p)
            testndx = fold(k,:);
        else
            tranndx(q,:) = fold(k,:);
            q = q+1;
        end
    end
    [u, w] = size (tranndx);
ntrain = w;
ntest = length(testndx);
trainndx = reshape(tranndx,1,(n-ntrain));
Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,(n-ntrain))');
Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest)');
 
 
ytrain = (mnist.train_labels(trainndx));
ytest  = (mnist.test_labels(testndx));
 
% Precompute sum of squares term for speed
XtrainSOS = sum(Xtrain.^2,2);
XtestSOS  = sum(Xtest.^2,2);
 
ypred = zeros(ntest,1);
 
% Classify
for i=1:ntest    
    dst = sqDistance(Xtest(i,:),Xtrain,XtestSOS(i,:),XtrainSOS);
    [junk,closest] = min(dst,[],2);
    ypred = ytrain(closest);
    if (ypred~=ytest(i,1))
        s = s+1;
    end
end
% Report
s;
errorRate = (s/1000);
end
s;
fprintf('Error Rate: %.2f%%\n',100*errorRate);
error(m,1)=errorRate*100;
% errorRate = mean(ypred ~= ytest);
end
plot (1:m,error)


