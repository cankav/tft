tft_clear();
randn('seed',0);

%% initialize test model data
movie_index = Index(177);
user_index = Index(480);
topic_index = Index(5000);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

X_data_orig = rand( movie_index.cardinality, user_index.cardinality );
Z1_data_orig = rand( topic_index.cardinality, movie_index.cardinality );
Z2_data_orig = rand( topic_index.cardinality, user_index.cardinality );

%X.data = X_data_orig;
Z1.data = Z1_data_orig;
Z2.data = Z2_data_orig;

% prepare base case result
X_dot_product = squeeze(Z2.data)' * squeeze(Z1.data);

pre_process();

mex -largeArrayDims CXXFLAGS='-std=c++11 -fPIC'  gtp_mex.cpp
%mex -largeArrayDims CXXFLAGS='-fPIC'  gtp_mex.cpp
gtp_mex(16, X, Z1, Z2);