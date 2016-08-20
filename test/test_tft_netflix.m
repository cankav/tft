clear all;
tft_clear();
rand('seed',0);

%% initialize test model data
movie_index = Index(17770);
user_index = Index(480189);
topic_index = Index(100);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

% initialize with random data, 1% sparsity
sparsity = 100480507 / ( 17770*480189 );
Z1.data = sparse( rand(topic_index.cardinality, movie_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, movie_index.cardinality);
Z2.data = sparse( rand(topic_index.cardinality, user_index.cardinality) > (1-sparsity) ) .* rand(topic_index.cardinality, user_index.cardinality);

pre_process();

%% gtp_mex sparse trial
gtp_mex_time = tic;
gtp_mex(16, X, Z1, Z2);
display( [ 'gtp_mex sparse trial time: ' num2str(toc(gtp_mex_time)) ] );