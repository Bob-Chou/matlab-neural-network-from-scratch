clear('all');
addpath('utils/');

% The checking approach is intuitive by using approximate numeral calculation
% Notice that grad(Y, X) = dY / dX, so we randomly choose a start X and a small
% delta value as dX, apply the forward calculation to get the Y and Y + dY, and
% calculate the grad = dY / dX. This grad should be very close to the output we
% call our backwards func (for example, less than delta)
delta = 1e-4;

disp("Check cross-entropy:")
A = rand(5, 10)+0.5;
B = (rand(5, 10) - 0.5);
[loss, cache_loss] = cross_entropy_forward(A, B);
dA = cross_entropy_backward(A, cache_loss);
dA_gap_max = 0;
for i = 1:5
    for j = 1:10
        A_n = zeros(5, 10);
        A_n(i, j) = delta;
        [loss_n, nc] = cross_entropy_forward(A+A_n, B);
        dAn = (loss_n - loss)/delta;
        dA_gap_max = max(dA_gap_max, abs(dAn-dA(i,j)));
    end
end
disp(["max gap of d(cross-entropy):", num2str(dA_gap_max)]);

disp("Check softmax:")
A = 10*(rand(5, 10) - 0.5);
B = 10*(rand(5, 10) - 0.5);
[probs, cache_softmax] = softmax_forward(A);
dA = softmax_backward(B, cache_softmax);
loss = sum(sum(B.*probs));
dA_gap_max = 0;
for i = 1:5
    for j = 1:10
        A_n = zeros(5, 10);
        A_n(i, j) = delta;
        [probs, nc] = softmax_forward(A+A_n);
        dAn = (sum(sum(B.*probs)) - loss)/delta;
        dA_gap_max = max(dA_gap_max, abs(dAn-dA(i,j)));
    end
end
disp(["max gap of d(Softmax):", num2str(dA_gap_max)]);

disp("Check ReLU:")
A = 10*(rand(5, 10) - 0.5);
B = 10*(rand(5, 10) - 0.5);
[y, cache_relu] = relu_forward(A);
dA = relu_backward(B, cache_relu);
loss = sum(sum(B.*y));
dA_gap_max = 0;
for i = 1:5
    for j = 1:10
        A_n = zeros(5, 10);
        A_n(i, j) = delta;
        [y, nc] = relu_forward(A+A_n);
        dAn = (sum(sum(B.*y)) - loss)/delta;
        dA_gap_max = max(dA_gap_max, abs(dAn-dA(i,j)));
    end
end
disp(["max gap of d(ReLu):", num2str(dA_gap_max)]);

disp("Check fc:")
A = 10*(rand(5, 5) - 0.5);
B = 10*(rand(1, 10) - 0.5);
C = 10*(rand(5, 10) - 0.5);
D = 10*(rand(5, 10) - 0.5);
[y, cache] = fc_forward(C, B, A);
[dW, db, dx] = fc_backward(D, cache);
loss = sum(sum(D.*y));
dC_gap_max = 0;
for i = 1:5
    for j = 1:10
        C_n = zeros(5, 10);
        C_n(i, j) = delta;
        [y, nc] = fc_forward(C+C_n, B, A);
        dCn = (sum(sum(D.*y)) - loss)/delta;
        dC_gap_max = max(dC_gap_max, abs(dCn-dW(i,j)));
    end
end
disp(["max gap of dW:", num2str(dC_gap_max)]);
dB_gap_max = 0;
for i = 1:1
    for j = 1:10
        B_n = zeros(1, 10);
        B_n(i, j) = delta;
        [y, nc] = fc_forward(C, B+B_n, A);
        dBn = (sum(sum(D.*y)) - loss)/delta;
        dB_gap_max = max(dB_gap_max, abs(dBn-db(i,j)));
    end
end
disp(["max gap of db:", num2str(dB_gap_max)]);
dA_gap_max = 0;
for i = 1:5
    for j = 1:5
        A_n = zeros(5, 5);
        A_n(i, j) = delta;
        [y, nc] = fc_forward(C, B, A+A_n);
        dAn = (sum(sum(D.*y)) - loss)/delta;
        dA_gap_max = max(dA_gap_max, abs(dAn-dx(i,j)));
    end
end
disp(["max gap of dX:", num2str(dA_gap_max)]);
