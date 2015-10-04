tft_clear();
randn('seed',0);

%% initialize test model data
movie_index = Index(177);
user_index = Index(480);
topic_index = Index(5000);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

Z1_data_orig = rand( topic_index.cardinality, movie_index.cardinality );
Z2_data_orig = rand( topic_index.cardinality, user_index.cardinality );

Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;

% prepare base case result
X_dot_product = squeeze(Z2.data)' * squeeze(Z1.data);

%% gtp_full trial
gtp_full_time = tic;
gtp_full(X, Z1, Z2);
display( [ 'gtp_full time: ' num2str(toc(gtp_full_time)) ] );

assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of full implementation and dot product are different.' );

%% reset trial data
global tft_indices 
tft_indices = []; % force pre_process to run again
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;

%% gtp trial
gtp_time = tic;
gtp(X, Z1, Z2);
display( [ 'gtp time: ' num2str(toc(gtp_time)) ] );
assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of standard implementation and dot product are different.' );

%% reset trial data
global tft_indices 
tft_indices = []; % force pre_process to run again
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;

%% gtp_mex trial
% pre_process();
% gtp_mex_time = tic;
% gtp_mex(X, Z1, Z2);
% display( [ 'gtp_mex time: ' num2str(toc(gtp_mex_time)) ] );
% assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of standard implementation and dot product are different.' );