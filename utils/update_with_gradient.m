function new_param= update_with_gradient(param, grad, lr)
    %======================================================
    % @ inputs:
    %   param: trainalbe parameter
    %   grad: gradient of this parameter
    %   lr: the learning rate
    % @ returns:
    %   new_param: new parameter
    %=======================================================
    new_param = param - lr*grad;
end
