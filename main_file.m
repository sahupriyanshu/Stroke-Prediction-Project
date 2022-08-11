
%% ===================== Soft Computing Assignment-1=====================
%
%
%% Data Set: Brain Stroke Prediction using Multi-variable Linear Regression
%
%
%% Clear and Close Figures
close all
clc

fprintf('Loading data ...\n');

%% Load Data
data = load('stroke_dataset.txt');
X = data(:, 1:9);
y = data(:, 10);
m = length(y);

% Print out some data points
fprintf('First 20 sample examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.0f \n', [X(1:20,:) y(1:20,:)]');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
%%
fprintf('\nNormalizing Features ...\n');

[X_Norm mu sigma] = feature_normalize(X);

% Add intercept term to X
X_Norm = [ones(m, 1) X_Norm];

x_train=X_Norm(1:4088,:);
y_train=y(1:4088,:);
x_test=X_Norm(4089:5110,:);
y_test=y(4089:5110,:);


%% Gradient Descent


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
num_iters = 3500;

% Init Theta and Run Gradient Descent 
theta = zeros(10, 1);
[theta, J_history] = gradient_descent_multi(x_train, y_train, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);
xlabel('Number of Iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('\nTheta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Predict the possibility of stroke with the given information of a person
% Formerly Smoked: 1 || Never Smoked: 0 || Smokes: 2 || Unknown: 3
% Rural: 0 || Urban: 1
% Male: 0 || Female: 1
% Yes: 1 || No: 0
% Gender: Female or 1, Age: 50, Hypertension: 1, Heart Disease: 0, Ever Married: Yes or 1, Residence: Urban, Glucose Level: 180, BMI: 30, Smoking: Formerly smoked 
stroke = [ 1 ( 1 -mu(1))/ sigma(1) (50 - mu(2))/sigma(2) (1 - mu(3))/sigma(3) (0 - mu(4))/ sigma(4) (1 - mu(5))/ sigma(5) (1 - mu(6))/ sigma(6) (180 - mu(7))/ sigma(7) (30 - mu(8))/ sigma(8) (1 - mu(9))/ sigma(9) ];
%%
fprintf(['\nPredicted possibility of stroke is  ' ...
         '(Using Gradient Descent):\n %f\n'], stroke);
     
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Error


J_train = compute_cost_multi(x_train, y_train, theta);
J_test = compute_cost_multi(x_test, y_test, theta);

fprintf('\nError in train dataset: %f \n', J_train);
fprintf('Error in test dataset: %f \n', J_test);



%% Regularization
alpha=0.1;
lambda=1;
iterations=3000;
theta=gradient_descent_reg(x_train,y_train,theta,alpha,iterations,lambda);

J_train = compute_cost_reg(x_train, y_train, theta,lambda);
J_test = compute_cost_reg(x_test, y_test, theta,lambda);


fprintf('\nWith regularization, Error in train dataset: %f \n', J_train);
fprintf('With regularization, Error in test dataset: %f \n', J_test);

%% Evaluation of R squared

hypo=x_train*theta;
err=(hypo-y_train).^2;
rss=sum(err);

y_avg=mean(y_train);
temp_var=(y_train-y_avg).^2;
tss=sum(temp_var);

r_squared=1-(rss/tss);
fprintf('\nR Squared Value = %f \n',r_squared);









