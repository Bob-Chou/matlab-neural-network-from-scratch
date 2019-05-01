function [dW1, db1, dW2, db2] = backward_propagation(probs, r_loss, r_probs, r_h2, r_relu1, r_h1)
    addpath('utils/');
    % cross-entropy loss
    dp = cross_entropy_backward(probs, r_loss);
    % softmax
    dl = softmax_backward(dp, r_probs);
    % layer 2
    %dh2 = relu_backward(dl, r_relu2);
    [dW2, db2, dx2] = fc_backward(dl, r_h2);
    % layer 1
    dx2 = relu_backward(dx2, r_relu1);
    [dW1, db1, nc] = fc_backward(dx2, r_h1);
end
