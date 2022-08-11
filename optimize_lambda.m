%% Clear and Close Figures
close all
clc

%% Load Data
fprintf('Loading data ...\n');
data=load('stroke_dataset.txt');

%% Splitting datasets

%train datasets
x_train=data(1:3000,1:9);
y_train=data(1:3000,10);

%cross validation 
xcv=data(3001:4088,1:9);
ycv=data(3001:4088,10);

%test datasets
x_test=data(4089:5110,1:9);
y_test=data(4089:5110,10);


m=length(x_train);
n=length(xcv);



%% Normalising
[x_train,mu,sigma]=feature_normalize(x_train);
[xcv,mu,sigma]=feature_normalize(xcv);
[x_test,mu,sigma]=feature_normalize(x_test);



%%  Hypothesis for different lambda
lambda=[1 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001];
x_train=[ones(m,1) x_train];
xcv=[ones(n,1) xcv];
J_cv=zeros(length(lambda),1);
alpha = 0.1;
num_iters = 5000;
for i=1:length(lambda)
    theta = zeros(10, 1);
    theta= gradient_descent_reg(x_train, y_train, theta, alpha, num_iters, lambda(i));
    J_cv(i)=compute_cost_multi(xcv,ycv,theta);
end

figure;

plot(1:numel(J_cv), J_cv, '-g', 'LineWidth', 2);
xlabel('lambda');
ylabel('Cost J');


[c,d]=find(J_cv==min(J_cv));
final_lambda=lambda(c);
fprintf('lambda determined is %f \n', final_lambda);

%% Hypothesis using obtained lambda

theta1=zeros(10,1);
theta1 = gradient_descent_reg(x_train, y_train, theta1, alpha, num_iters, final_lambda);
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta1);
fprintf('\n');

%% J_test and J_cv
l=length(x_test);
x_test=[ones(l,1) x_test];
J_test=compute_cost_multi(x_test,y_test,theta);
fprintf('Test error = %f \n',J_test);
J_cv=compute_cost_multi(xcv,ycv,theta);
fprintf('Cross validation error = %f \n',J_cv);
