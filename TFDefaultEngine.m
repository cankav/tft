classdef TFDefaultEngine < handle
    properties
        kl_divergence;
        gtp_implementation_selection;
        config;
    end

    methods
        function obj = TFDefaultEngine(config, gtp_implementation_selection)
            obj.config = config;
            assert( isstr(gtp_implementation_selection) && (strcmp(gtp_implementation_selection, 'gtp') || strcmp(gtp_implementation_selection, 'gtp_mex') || strcmp(gtp_implementation_selection, 'gtp_full')), 'TFEngineConfig:TFEngineConfig', 'gtp implementation selection must be a string containing one of the following ''gtp'', ''gtp_full'', ''gtp_mex''' );
            obj.gtp_implementation_selection = gtp_implementation_selection;

            obj.kl_divergence = zeros(size(config.tfmodel.coupling_matrix,1), config.iteration_number);
        end

        function [] = factorize(obj)
            total_tic = tic;
            execution_times = zeros(length(obj.config.gtp_rules),1);
            for it_num = 1:obj.config.iteration_number
                iteration_tic = tic;
                display([ char(10) 'iteration ' num2str(it_num) ]);
                for rule_ind = 1:length(obj.config.gtp_rules)
                    % if rule_ind == 10 || rule_ind == 18 || rule_ind == 26
                    %     dbstop TFDefaultEngine 25
                    % end

                    % if iscell( obj.config.gtp_rules{rule_ind}{3} )
                    %     input = num2str(cellfun( @(x) x.id, obj.config.gtp_rules{rule_ind}{3} ));
                    % else
                    %     input = obj.config.gtp_rules{rule_ind}{3};
                    % end
                    display_rule( obj.config.gtp_rules{rule_ind}, rule_ind, 'Executing ' );

                    %for ri = 1:length(obj.config.gtp_rules)
                    %    display_rule( obj.config.gtp_rules{ri}, ri, 'ALL RULES ' );
                    %end

                    execution_tic = tic;
                    if obj.config.gtp_rules{rule_ind}{1} == 'GTP'
                        assert( sum_all_dims(size(obj.config.gtp_rules{rule_ind}{2}.data)) ~= 0, 'TFDefaultEngine:TFDefaultEngine', 'GTP operation requires output tensor with non-zero data' );
                        input_tensors = obj.config.gtp_rules{rule_ind}{3};
                        if strcmp(obj.gtp_implementation_selection, 'gtp_mex')
                            gtp_mex(16, obj.config.gtp_rules{rule_ind}{2}, input_tensors{:} );
                        elseif strcmp(obj.gtp_implementation_selection, 'gtp')
                            gtp(obj.config.gtp_rules{rule_ind}{2}, input_tensors{:} );
                        elseif strcmp(obj.gtp_implementation_selection, 'gtp_full')
                            gtp_full(obj.config.gtp_rules{rule_ind}{2}, input_tensors{:} );
                        end
                    else
                        obj.config.gtp_rules{rule_ind}{2}.data = eval( obj.config.gtp_rules{rule_ind}{3} );
                    end
                    execution_times(rule_ind) = execution_times(rule_ind) + toc(execution_tic);
                end

                obj.kl_divergence( :, it_num ) = get_kl_divergence_values(obj.config.tfmodel);

                display( ['iteration time ' num2str(toc(iteration_tic)) ' seconds divergences ' regexprep(num2str( obj.kl_divergence( :, it_num ) ), '\s*', ',') ] );

            end % end iteration
            display( ['total time ' num2str(toc(total_tic)) ' average execution_times ' num2str((execution_times./obj.config.iteration_number)')] );
        end
    end
end