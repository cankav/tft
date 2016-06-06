clear all;

tft_clear();
rand('seed',0);

%dbstop TFDefaultEngine 44

test_model_type = 'tucker3_nmf';

generate_test_model(test_model_type);

model = TFModel(factorization_model, p, phi);

gtp_rules = model.update_rules()

test_gtp_modes = {'gtp_full', 'gtp', 'gtp_mex'};
for i = 1:length(test_gtp_modes)
    display(['testing in ' test_gtp_modes{i} ' mode']);
    config = TFEngineConfig(model, 10);
    engine = TFDefaultEngine(config, test_gtp_modes{i});

    % for rule_ind = 1:length(config.gtp_rules)
    %     display_rule( config.gtp_rules{rule_ind}, rule_ind, 'Executing ' );
    % end
    
    engine.factorize();
    plot(engine.kl_divergence');
    pause(3);
    break
end
