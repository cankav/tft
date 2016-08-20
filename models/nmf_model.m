movie_index = Index(17);
user_index = Index(4);
topic_index = Index(10);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

X.data = rand(movie_index.cardinality, user_index.cardinality);
Z1.data = rand( topic_index.cardinality, movie_index.cardinality );
Z2.data = rand( topic_index.cardinality, user_index.cardinality );

pre_process();

p = [1];
phi = [1];

factorization_model = {X, {Z1, Z2}};