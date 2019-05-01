function [y, cache] = relu_forward(x)
    %======================================================
    % @ inputs:
    %   x: the data, should be a matrix in [batch, dimenson] format
    % @ returns:
    %   y: a matrix identical with relu(x)
    %   cache: cache from forward
    %=======================================================
    y = max(x, 0);
    cache = y;
end
