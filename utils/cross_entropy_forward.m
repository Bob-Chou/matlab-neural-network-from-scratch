function [loss, cache] = cross_entropy_forward(probs, y)
    %======================================================
    % @ inputs:
    %   probs: a matrix recording the probs of each category, in format [batch, categories]
    %   y: a matrix recording the labels of input data, in format [batch, categories]
    % @ returns:
    %   loss: the total loss computed by cross entropy
    %   cache: intermediate results to facilitate backward computing
    % @ Hint:
    %   loss should be sumed up to a scalar
    %=======================================================
    loss = -sum(sum(y.*log(probs)));
    cache = y;
end
