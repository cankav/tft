tft_clear();
randn('seed',0);

movie_index = Index(177);
user_index = Index(480);
topic_index = Index(5000);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

X_data_orig = rand( movie_index.cardinality, user_index.cardinality );
Z1_data_orig = rand( topic_index.cardinality, movie_index.cardinality );
Z2_data_orig = rand( topic_index.cardinality, user_index.cardinality );

X.data = X_data_orig;
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;

gtp_full_time = tic;
gtp_full(X, Z1, Z2);
display( [ 'gtp_full time: ' num2str(toc(gtp_full_time)) ] );

X_dot_product = squeeze(Z2.data) * squeeze(Z1.data)';
assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of full implementation and dot product are different.' );

global tft_indices
tft_indices = [];
X.data = X_data_orig;
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;

gtp_time = tic;
gtp(X, Z1, Z2);
display( [ 'gtp time: ' num2str(toc(gtp_time)) ] );

assert( sum_all_dims( float_diff(X_dot_product', squeeze(X.data)) ) == 0, 'test_tft:test_tft', 'Result of standard implementation and dot product are different.' );

% TODO: test with matlab dot product

%gtp_mex_time = tic;
%gtp_mex(X, Z1, Z2);
%display( [ 'gtp_mex time: ' num2str(toc(gtp_mex_time)) ] );
