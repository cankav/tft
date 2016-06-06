function [] = test_gctf_steiner_helper(test_model_type, gtp_rule_group_ids, base_filename)
    evalin('base','clear all');
    tft_clear();
    rand('seed',0);

    generate_test_model(test_model_type);

    evalin('base', 'model = TFModel(factorization_model, p, phi);');

    gtp_rules = evalin('base', 'model.update_rules();');

    for rule_ind = 1:length(gtp_rules)
        display_rule( gtp_rules{rule_ind}, rule_ind, 'rule ' );
    end

    evalin('base', 'config = TFEngineConfig(model, 10);');
    if nargin==3
        evalin('base', ['engine = TFSteinerEngine(config, [' num2str(gtp_rule_group_ids) '], ''' base_filename ''');']);
    else
        evalin('base', ['engine = TFSteinerEngine(config, [' num2str(gtp_rule_group_ids) '] );']);
    end

    evalin('base', 'engine.factorize();');
    figure
    evalin('base', 'plot(engine.kl_divergence'');');
end