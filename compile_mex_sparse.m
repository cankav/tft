clear all;
tft_clear();
randn('seed',0);

%% initialize test model data
movie_index = Index(177);
user_index = Index(480);
topic_index = Index(5000);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

% initialize with random data, 1% sparsity
sparsity = 0.01;
Z1.data = sparse( rand(topic_index.cardinality, movie_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, movie_index.cardinality);
Z2.data = sparse( rand(topic_index.cardinality, user_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, user_index.cardinality);

% prepare base case result
X_dot_product = Z2.data' * Z1.data;

pre_process();

%Z1.data
%nnz(Z1.data)
%nnz(Z1.data) / numel(Z1.data)
%Z2.data
%nnz(Z2.data)
%nnz(Z2.data) / numel(Z2.data)

% fpermissive is required to conform with gtp(X, Z1, Z2) syntax, otherwise syntax must be X=gtp(Z1,Z2)
mex -largeArrayDims CXXFLAGS='-std=c++11 -fPIC -fpermissive'  gtp_mex.cpp % c++11 for print mutex lock
%mex -largeArrayDims CXXFLAGS='-fPIC -fpermissive'  gtp_mex.cpp
gtp_mex_time = tic;
gtp_mex(1, X, Z1, Z2); % TODO: how to implement parallel output_irs write? %%% IF assume output full -> can run parallel
display( [ 'gtp_mex time: ' num2str(toc(gtp_mex_time)) ] );

assert( sum_all_dims( float_diff( reshape(X_dot_product', [prod(size(X_dot_product)), 1]), squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of standard implementation and dot product are different.' );