 function [theta, J_history] = gradient_descent_multi(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
m=length(y);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters

    Jn=0;
    for k=1:1:length(X(1,:))
        h=0;
        Jn=0;
        h=X*theta;
    for i=1:1:m
        Jn=Jn+((h(i)-y(i)))*X(i,k);
    end
        J(k)=Jn/m;
        temp_var(k)=theta(k)-alpha*J(k);
    end

    for k=1:1:length(X(1,:))
        theta(k)=temp_var(k);
    end
        % Save the cost J in every iteration    
    J_history(iter) = compute_cost_multi(X, y, theta);
end
end


