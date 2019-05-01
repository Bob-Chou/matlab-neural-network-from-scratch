function [loss, probs, r_h1, r_relu1, r_h2, r_probs, r_loss] = forward_propagation(x, y, W1, b1, W2, b2)
    addpath('utils/');
    % layer 1
    [h1, r_h1] = fc_forward(W1, b1, x);
    [relu1, r_relu1] = relu_forward(h1);
    % layer 2
    [h2, r_h2] = fc_forward(W2, b2, relu1);
    % softmax
    [probs, r_probs] = softmax_forward(h2);
    % cross-entropy loss
    [loss, r_loss] = cross_entropy_forward(probs, y);
end
