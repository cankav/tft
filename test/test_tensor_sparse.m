clear all
addpath('/home/can/projeler/tft');
setup_tft('/home/can/projeler/tft');

index_i = Index(10);
index_j = Index(20);
index_k = Index(30);

X = Tensor(index_i, index_j, index_k);

X.data = sparse(10*20*30,1);

pre_process(); % must be called after setting data

% get first element of X
assert( X(1) == X{1,1,1} )

% TODO add sparse examples
% TODO add examples not utilizing all dimensions