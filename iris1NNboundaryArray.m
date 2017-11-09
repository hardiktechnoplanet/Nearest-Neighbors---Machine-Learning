pkg load statistics

% setup data
D = csvread('iris.csv');
X_train = D(:, 1:2);
y_train = D(:, end);
error=0;
T = [10,20,30,50];
E = zeros(1,5);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for q=1:4
for m=1:4
X = randi(150,[T(m),1]);
%for q=1:10
%  P(q) =X(q);
%end

for j=1:T(m)
  
  if((y_train(X(j),1)==1)||(y_train(X(j),1)==2))
     y_train(X(j),1)=y_train(X(j),1)+1;
  else
     y_train(X(j),1)=y_train(X(j),1)-1;
  end
end
%end
%y_train = y_train+1;
% setup meshgrid
[x1, x2] = meshgrid(2:0.01:5, 0:0.01:3);
grid_size = size(x1);
X12 = [x1(:) x2(:)];

% compute 1NN decision 
n_X12 = size(X12, 1);
decision = zeros(n_X12, 1);
for i=1:n_X12    
    point = X12(i, :);
    
    % compute euclidan distance from the point to all training data
    dist = pdist2(X_train, point);
    
    % sort the distance, get the index
    [~, idx_sorted] = sort(dist);
    
    % find the class of the nearest neighbour
    pred = y_train(idx_sorted(3));
    
    decision(i) = pred;
end

% plot decisions in the grid
%errorRate = mean(ypred ~= ytest);
%E(j)=errorRate;
%disp(E);
figure
decisionmap = reshape(decision, grid_size);
imagesc(2:0.01:5, 0:0.01:3, decisionmap);
set(gca,'ydir','normal');

% colormap for the classes
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1];
colormap(cmap);

% satter plot data

hold on;
scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 10, 'r');
scatter(X_train(y_train == 2, 1), X_train(y_train == 2, 2), 10, 'g');
scatter(X_train(y_train == 3, 1), X_train(y_train == 3, 2), 10, 'b');
hold off;
%
%endfor
%this for loop is used to find the error
for j=1:T(m)
  if (y_train(X(j),1)~=decision(X(j),1))
    error=error+1;
    end
endfor
display(error);
%E(j)=error;
%display(E);
error=0;
endfor