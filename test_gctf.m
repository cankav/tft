clear all;
tft_clear();
rand('seed',0);

%% initialize test model data
movie_index = Index(177);
user_index = Index(480);
topic_index = Index(1000);

X = Tensor( movie_index, user_index );
Z1 = Tensor( topic_index, movie_index);
Z2 = Tensor( topic_index, user_index );

X.data = rand(movie_index.cardinality, user_index.cardinality); % observed tensor data
Z1.data = rand( topic_index.cardinality, movie_index.cardinality ); % randomly initialize latent tensors
Z2.data = rand( topic_index.cardinality, user_index.cardinality ); 
pre_process();

p = [1]; % for KL divergence
phi = [1]; % dispersion parameter
factorization_model = {X, {Z1, Z2}}; % factorization model

nmf_model = TFModel(factorization_model, p, phi);

% generate GTP operations for GCTF update rules
gtp_rules = nmf_model.update_rules()

test_gtp_modes = {'gtp_full', 'gtp', 'gtp_mex'};
for i = 1:length(test_gtp_modes)
    display(['testing in ' test_gtp_modes{i} ' mode']);
    % apply update rule GTP operations 10 times, without any optimizations
    config = TFEngineConfig(nmf_model, 10);
    engine = TFDefaultEngine(config, test_gtp_modes{i});
    engine.factorize();
    plot(engine.kl_divergence);
end
