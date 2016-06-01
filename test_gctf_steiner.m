clear all;
tft_clear();
rand('seed',0);

test_model_type = 'nmf';

generate_test_model(test_model_type);

model = TFModel(factorization_model, p, phi);

gtp_rules = model.update_rules();

config = TFEngineConfig(model, 10);
engine = TFSteinerEngine(config, [ 1, -1, 1, -1, -1, 1, -1, -1, 2, -1, 2, -1, -1, 2, -1, -1 ], 'steiner_test');
engine.factorize();
figure
plot(engine.kl_divergence);
