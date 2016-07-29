clear all;

tft_clear();
rand('seed',0);

movie_index = Index(131262);
user_index = Index(138493);
genre_index = Index(19);

X = Tensor( user_index, movie_index );
Z1 = Tensor( genre_index, movie_index);
Z2 = Tensor( genre_index, user_index );

load data/movielens/ml-20m/ratings.mat
X.data = ratings;
Z1.data = rand( genre_index.cardinality, movie_index.cardinality );
Z2.data = rand( genre_index.cardinality, user_index.cardinality );

pre_process();

p = [1];
phi = [1];

factorization_model = {X, {Z1, Z2}};

model = TFModel(factorization_model, p, phi);

gtp_rules = model.update_rules();

for rule_ind = 1:length(gtp_rules)
    display_rule( gtp_rules{rule_ind}, rule_ind, 'rule ' );
end

config = TFEngineConfig(model, 10);
engine = TFDefaultEngine(config, 'gtp_mex');
engine.factorize();
plot(engine.kl_divergence'');
check_divergence(engine.kl_divergence);
