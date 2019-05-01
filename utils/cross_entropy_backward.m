function dp = cross_entropy_backward(probs, cache)
    %======================================================
    % @ inputs:
    %   probs: probs from the output of forward, in format [batch, categories]
    %   cache: cache from forward
    % @ returns:
    %   dp: gradient of probs to downstream, a matrix in [batch, categories] format
    %=======================================================
    dp = -cache./probs;
end
