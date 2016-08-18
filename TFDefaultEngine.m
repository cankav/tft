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
            % TODO: check expected data sizes before execution

            factorize_start_clock = clock;
            factorize_timeout = false;
            total_tic = tic;
            execution_times = zeros(length(obj.config.gtp_rules),1);
            for it_num = 1:obj.config.iteration_number
                iteration_tic = tic;
                %display([ char(10) char(10) char(10) 'iteration ' num2str(it_num) ]);
                for rule_ind = 1:length(obj.config.gtp_rules)
                    % if rule_ind == 8
                    %     display('before rule');
                    %     evalin('base', 'display(A.data(1:10))');
                    % elseif rule_ind == 16
                    %     display('before rule');
                    %     evalin('base', 'display(B.data(1:10))');
                    % elseif rule_ind == 24
                    %     display('before rule');
                    %     evalin('base', 'display(C.data(1:10))');
                    % elseif rule_ind == 32
                    %     display('before rule');
                    %     evalin('base', 'display(G.data(1:10))');
                    % end
                                        
                    % if rule_ind == 3 % || rule_ind == 18 || rule_ind == 26
                    %     dbstop TFDefaultEngine 58
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
                        %display('before gtp');
                        % before = obj.config.gtp_rules{rule_ind}{2}.data;
                        %display(obj.config.gtp_rules{rule_ind}{2}.data(1:10));
                        assert( sum_all_dims(size(obj.config.gtp_rules{rule_ind}{2}.data)) ~= 0, 'TFDefaultEngine:TFDefaultEngine', 'GTP operation requires output tensor with non-zero data' );
                        input_tensors = obj.config.gtp_rules{rule_ind}{3};
                        if strcmp(obj.gtp_implementation_selection, 'gtp_mex')
                            gtp_mex(8, obj.config.gtp_rules{rule_ind}{2}, input_tensors{:} );
                        elseif strcmp(obj.gtp_implementation_selection, 'gtp')
                            gtp(obj.config.gtp_rules{rule_ind}{2}, input_tensors{:} );
                        elseif strcmp(obj.gtp_implementation_selection, 'gtp_full')
                            gtp_full(obj.config.gtp_rules{rule_ind}{2}, input_tensors{:} );
                        end

                        %display('after gtp');
                        %display(obj.config.gtp_rules{rule_ind}{2}.data(1:10));
                        %display([ 'all the same ' num2str( sum_all_dims( before == obj.config.gtp_rules{rule_ind}{2}.data ) == numel(before) ) ' numel ' num2str(numel(before)) ' diff num ' num2str(sum_all_dims( before ~= obj.config.gtp_rules{rule_ind}{2}.data ) ) ]);
                    else
                        % if rule_ind == 32
                        %     dbstop TFDefaultEngine 75
                        % end

                        obj.config.gtp_rules{rule_ind}{2}.data = eval( obj.config.gtp_rules{rule_ind}{3} );
                    end
                    execution_times(rule_ind) = execution_times(rule_ind) + toc(execution_tic);

                    % if rule_ind == 8
                    %     display('after rule');
                    %     evalin('base', 'display(A.data(1:10))');
                    % elseif rule_ind == 16
                    %     display('after rule');
                    %     evalin('base', 'display(B.data(1:10))');
                    % elseif rule_ind == 24
                    %     display('after rule');
                    %     evalin('base', 'display(C.data(1:10))');
                    % elseif rule_ind == 32
                    %     display('after rule');
                    %     evalin('base', 'display(G.data(1:10))');
                    % end

                    if obj.config.timeout ~= 0 && etime(clock, factorize_start_clock) > obj.config.timeout
                        display(['Warning: Factorization timeout at iteration: ' num2str(it_num) ' rule_ind ' num2str(rule_ind)]);
                        factorize_timeout = true;
                        break
                    end

                end % end rule_ind

                if factorize_timeout
                    break
                end

                obj.kl_divergence( :, it_num ) = get_kl_divergence_values(obj.config.tfmodel);

                %display( ['iteration time ' num2str(toc(iteration_tic)) ' seconds divergences ' num2str( obj.kl_divergence( :, it_num )' ) ] );

            end % end iteration
            display( ['total time ' num2str(toc(total_tic)) ' average execution_times ' num2str((execution_times./obj.config.iteration_number)')] );
        end
    end
end