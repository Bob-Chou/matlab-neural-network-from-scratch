clear('all');
addpath('utils/');
addpath('lib/');

%===========================================
%
%       Define Hyper-parameters
%
%===========================================
num_categories = 10;
num_hidden_dims = 32;
num_image_dims = 28 * 28;

lr = 1e-2;
decay_ratio = 0.1;
decay_step = 1000;
batch_size = 32;
total_step = 5000;
init_eps = 0.08;

print_every = 100;
test_num = 1000;
%===========================================
%
%       Read Data
%
%===========================================
train_x = read_mnist_data("./MNIST/train-images-idx3-ubyte");
train_y = read_mnist_label("./MNIST/train-labels-idx1-ubyte");
test_x = read_mnist_data("./MNIST/t10k-images-idx3-ubyte");
test_y = read_mnist_label("./MNIST/t10k-labels-idx1-ubyte");
% One-hot encoding
train_y(train_y==0) = 10;
train_y = eye(max(train_y))(train_y,:);
test_y(test_y==0) = 10;
test_y = eye(max(test_y))(test_y,:);
% Fetch fixed test data
fix_idx = randperm(length(test_y));
fix_x = test_x(fix_idx, :)(1:test_num,:);
fix_y = test_y(fix_idx, :)(1:test_num,:);
%===========================================
%
%       Initialize Parameters
%
%===========================================
W1 = rand(num_image_dims, num_hidden_dims) * 2 * init_eps - init_eps;
W2 = rand(num_hidden_dims, num_categories) * 2 * init_eps - init_eps;
b1 = ones(1, num_hidden_dims);
b2 = ones(1, num_categories);

start_idx = 1;
end_idx = 1;
idxs = randperm(length(train_y));
train_x = train_x(idxs, :);
train_y = train_y(idxs, :);

start_idx = end_idx;
end_idx = min(start_idx+batch_size, length(train_y)+1);
x = train_x(start_idx:end_idx-1, :);
y = train_y(start_idx:end_idx-1, :);

for iter = 1:total_step
    if mod(iter, decay_step) == 0
        lr = lr * decay_ratio;
    end
    %===========================================
    %
    %       Get minibatch
    %
    %===========================================
    start_idx = end_idx;
    end_idx = min(start_idx+batch_size, length(train_y)+1);
    x = train_x(start_idx:end_idx-1, :);
    y = train_y(start_idx:end_idx-1, :);
    %===========================================
    %
    %       Forward
    %
    %===========================================
    [loss, probs, cache_h1, cache_relu1, cache_h2, cache_probs, cache_loss] = forward_propagation(x, y, W1, b1, W2, b2);
    if mod(iter, print_every) == 0
        [nc, train_infer] = max(probs,[],2);
        [nc, train_true] = max(y,[],2);
        [nc, test_probs] = forward_propagation(fix_x, fix_y, W1, b1, W2, b2);
        [nc, test_infer] = max(test_probs,[],2);
        [nc, test_true] = max(fix_y,[],2);
        disp(["step: ", num2str(iter), " average loss: ", num2str(loss/(end_idx-start_idx-1)), " train acc: ", num2str(sum(train_infer==train_true)/length(train_true)), " test acc: ", num2str(sum(test_infer==test_true)/length(test_true))]);
    end
    %===========================================
    %
    %       Backward
    %
    %===========================================
    [dW1, db1, dW2, db2] = backward_propagation(probs, cache_loss, cache_probs, cache_h2, cache_relu1, cache_h1);

    %===========================================
    %
    %       Learning
    %
    %===========================================
    W1 = update_with_gradient(W1, dW1, lr);
    b1 = update_with_gradient(b1, db1, lr);
    W2 = update_with_gradient(W2, dW2, lr);
    b2 = update_with_gradient(b2, db2, lr);

    if end_idx >= length(train_y)
        start_idx = 1;
        end_idx = 1;
        idxs = randperm(length(train_y));
        train_x = train_x(idxs, :);
        train_y = train_y(idxs, :);
    end
end
