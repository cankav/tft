clear all;
tft_clear();
rand('seed',0);
global tft_indices

%% initialize test model data
movie_index = Index(177);
user_index = Index(480);
topic_index = Index(1000);

X = Tensor( movie_index, user_index );

X.data = rand(movie_index.cardinality, user_index.cardinality);

Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

p = [1]; % for KL divergence
phi = [1]; % dispersion parameter
factorization_model = {X, {Z1, Z2}}; % factorization model

nmf_model = TFModel(factorization_model, p, phi)

% generate GTP operations for GCTF update rules
gtp_rules = nmf_model.update_rules()

% % apply update rule GTP operations 10 times, without any optimizations
% engine = DefaultEngine(nmf_model_gctf_gtps, 10)
% engine.factorize();
% plot(engine.kl_divergence_values);

% % apply update rule GTP operations 10 times, via optimal path in steiner tree
% engine = SteinerEngine(nmf_model_gctf_gtps, 10)
% engine.factorize();
% plot(engine.kl_divergence_values);