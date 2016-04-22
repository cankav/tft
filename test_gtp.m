clear all;
tft_clear();
rand('seed',0);
global tft_indices

%% initialize test model data
movie_index = Index(177);
user_index = Index(480);
topic_index = Index(1000);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

% necessary for running pre_process multiple times in single environment
Z1_data_orig = rand( topic_index.cardinality, movie_index.cardinality );
Z2_data_orig = rand( topic_index.cardinality, user_index.cardinality );
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;

% prepare base case result
X_dot_product = Z2.data' * Z1.data;

%% gtp_full trial
gtp_full_time = tic;
gtp_full(X, Z1, Z2);
display( [ 'gtp_full trial time: ' num2str(toc(gtp_full_time)) char(10)] );
assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of full implementation and dot product are different.' );

%% gtp trial
tft_indices = [];
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;
gtp_time = tic;
gtp(X, Z1, Z2);
display( [ 'gtp standard trial time: ' num2str(toc(gtp_time)) char(10)] );
assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of standard implementation and dot product are different.' );

%% gtp_mex dense trial
tft_indices = [];
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;
pre_process();
gtp_mex_time = tic;
gtp_mex(16, X, Z1, Z2);
display( [ 'gtp_mex dense trial time: ' num2str(toc(gtp_mex_time)) char(10)] );
assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of dense mex implementation and dot product are different.' );

%% gtp_mex sparse trial
% initialize with random data, 1% sparsity
sparsity = 0.01;
Z1.data = sparse( rand(topic_index.cardinality, movie_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, movie_index.cardinality);
Z2.data = sparse( rand(topic_index.cardinality, user_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, user_index.cardinality);
X_dot_product = Z2.data' * Z1.data;
tft_indices = [];
pre_process();
gtp_mex_time = tic;
gtp_mex(16, X, Z1, Z2);
display( [ 'gtp_mex sparse trial time: ' num2str(toc(gtp_mex_time)) ] );
assert( sum_all_dims( float_diff( X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of sparse mex implementation and dot product are different.' );