function dl = softmax_backward(dp, cache)
    %======================================================
    % @ inputs:
    %   dp: gradient of probs from upstream, a matrix in [batch, categories] format
    %   cache: cache from forward
    % @ returns:
    %   dl: gradient of logits to downstream, a matrix in [batch, categories] format
    % @ Hint:
    %   You'd better write matrix-wise code
    %=======================================================
    dl = cache.*dp-cache.*sum(cache.*dp, 2);
end
