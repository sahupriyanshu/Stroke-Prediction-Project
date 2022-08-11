function theta =gradient_descent_reg(X,Y,theta,alpha,iterations,lambda)

m=length(Y);

for i=1:iterations
     theta=theta*(1-alpha*lambda/m)-(X'*(X*theta-Y))*alpha/m;
end
end