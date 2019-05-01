function [probs, cache] = softmax_forward(logits)
    %======================================================
    % @ inputs:
    %   logits: the data, should be a matrix in [batch, categories] format
    % @ returns:
    %   probs: the ouput probabilites derived by softmax with regards to logits.
    %   cache: intermediate results to facilitate backward computing
    % @ Hint:
    %   probs = exp(logits[single_category])/sum(exp(logits))
    %=======================================================

end
