clear all;
tft_clear();
rand('seed',0);

%% initialize test model data
movie_index = Index(17);
user_index = Index(48);
topic_index = Index(10);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

% initialize with random data, 50% sparsity
sparsity = 0.5;
Z1.data = sparse( rand(topic_index.cardinality, movie_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, movie_index.cardinality);
Z2.data = sparse( rand(topic_index.cardinality, user_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, user_index.cardinality);

% prepare base case result
X_dot_product = Z2.data' * Z1.data;

X.data = reshape(X_dot_product', prod(size(X_dot_product)), 1); %(X_dot_product'~=0) = 1;

first_nnz_ind = find(X.data, 1, 'first');
X.data(first_nnz_ind) = rand();

pre_process();

% fpermissive is required to conform with gtp(X, Z1, Z2) syntax,
% otherwise syntax must be X=gtp(Z1,Z2)
% compile with -DTFT_DEBUG for debug output
mex -outdir core -largeArrayDims CXXFLAGS='-O3 -fPIC -fpermissive -std=c++11' core/gtp_mex.cpp
gtp_mex_time = tic;
gtp_mex(8, X, Z1, Z2);
display( [ 'gtp_mex time: ' num2str(toc(gtp_mex_time)) ] );

assert( sum_all_dims( float_diff(reshape(X_dot_product', prod(size(X_dot_product)), 1), X.data) ) == 0, 'compile_mex_sparse:compile_mex_sparse', 'Result of standard implementation and dot product are different.' );
