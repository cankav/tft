function [] = test_gctf_helper(test_model_type)
    evalin('base', 'clear all;');
    tft_clear();
    rand('seed',0);

    generate_test_model(test_model_type);

    evalin('base', 'model = TFModel(factorization_model, p, phi);');

    gtp_rules = evalin('base', 'model.update_rules();');

    for rule_ind = 1:length(gtp_rules)
        display_rule( gtp_rules{rule_ind}, rule_ind, 'rule ' );
    end

    test_gtp_modes = {'gtp_full', 'gtp', 'gtp_mex'};
    for i = 1:length(test_gtp_modes)
        display(['testing in ' test_gtp_modes{i} ' mode']);
        evalin('base', 'config = TFEngineConfig(model, 10);');
        evalin('base', ['engine = TFDefaultEngine(config, ''' test_gtp_modes{i} ''');']);
        evalin('base', 'engine.factorize();');
        evalin('base', 'plot(engine.kl_divergence'');');
        pause(3);
    end

end