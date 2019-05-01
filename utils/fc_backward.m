function [dW, db, dx] = fc_backward(dy, cache)
    %======================================================
    % @ inputs:
    %   dy: gradient of y from upstream, a matrix in [batch, dimension] format
    %   cache: cache from forward
    % @ returns:
    %   dW: gradient of W to downstream, a matrix in [dimension, output] format
    %   db: gradient of b to downstream, a vector
    %   dx: gradient of x to downstream, a matrix in [batch, dimension] format
    %=======================================================
    dW = (cache.x)'*dy;
    dx = dy*(cache.w)';
    db = sum(dy, 1);
end
