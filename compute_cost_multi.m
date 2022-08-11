function J=compute_cost_multi(X,y,theta)
m=length(y);
h=X*theta;
err=(h-y).^2;
J=(1/(2*m))*sum(err);
end