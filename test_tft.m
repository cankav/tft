tft_clear();
randn('seed',0);

movie_index = Index(177);
user_index = Index(480);
topic_index = Index(5000);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );


X.data = rand( movie_index.cardinality, user_index.cardinality );
Z1.data = rand( topic_index.cardinality, movie_index.cardinality );
Z2.data = rand( topic_index.cardinality, user_index.cardinality );

gtp_full_time = tic;
gtp_full(X, Z1, Z2)
display( [ 'gtp_full time: ' num2str(toc(gtp_full_time)) ] );