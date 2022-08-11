function [X, mu, sigma] = feature_normalize(X)

 mu=mean(X)
 sigma=std(X)
 [m,n]=size(X)
 
 for j=1:1:n
     X(:,j)= (X(:,j)-mu(j))/sigma(j);
 end

% ============================================================

end
