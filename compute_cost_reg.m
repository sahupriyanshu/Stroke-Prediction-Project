function J=compute_cost_reg(X,Y,theta,lambda)

m=length(Y);

J=(sum((X*theta-Y).^2)/(2*m))+sum(theta.^2)*lambda/(2*m);
end