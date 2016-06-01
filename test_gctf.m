clear all;

tft_clear();
rand('seed',0);

%dbstop TFDefaultEngine 21

test_model_type = 'tucker3';

generate_test_model(test_model_type);

model = TFModel(factorization_model, p, phi);

gtp_rules = model.update_rules()

test_gtp_modes = {'gtp_full', 'gtp', 'gtp_mex'};
for i = 1:length(test_gtp_modes)
    i=3
    display(['testing in ' test_gtp_modes{i} ' mode']);
    config = TFEngineConfig(model, 10);
    engine = TFDefaultEngine(config, test_gtp_modes{i});
    engine.factorize();
    plot(engine.kl_divergence);
    
    break
end
