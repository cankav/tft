clear all;
tft_clear();
rand('seed',0);

%% initialize test model data
movie_index = Index(7);
user_index = Index(4);
topic_index = Index(10);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

Z1.data = rand( topic_index.cardinality, movie_index.cardinality );
Z2.data = rand( topic_index.cardinality, user_index.cardinality );

% prepare base case result
%x_dot_product_time = tic;
X_dot_product = Z2.data' * Z1.data;
%display( [ 'matlab product time: ' num2str(toc(x_dot_product_time)) ] );

pre_process();

% fpermissive is required to conform with gtp(X, Z1, Z2) syntax, otherwise syntax must be X=gtp(Z1,Z2)
% compile with -DTFT_DEBUG for debug output
mex -outdir core -largeArrayDims CXXFLAGS='-O3 -std=c++11 -fPIC -fpermissive' core/gtp_mex.cpp
gtp_mex_time = tic;
gtp_mex(8, X, Z1, Z2);
display( [ 'gtp_mex time: ' num2str(toc(gtp_mex_time)) ] );

assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of standard implementation and dot product are different.' );