classdef TFDefaultEngine < handle
    properties
        gtp_rules;
        tfmodel;
        iteration_number;
        kl_divergence;
    end

    methods
        function obj = TFDefaultEngine(varargin)
            assert( length(varargin) == 2, 'TFDefaultEngine:TFDefaultEngine', 'TFDefaultEngine needs exactly 2 arguments: factorization_model, number of iterations' );
            assert( isa(varargin{1}, 'TFModel'), 'TFDefaultEngine:TFDefaultEngine', 'First argument must be of an instance of TFModel' );
            assert( isnumeric(varargin{2}) && length(varargin{1})==1, 'TFDefaultEngine:TFDefaultEngine', 'Iteration number must be a single integer' );

            obj.tfmodel = varargin{1};
            obj.iteration_number = varargin{2};

            assert( sum(obj.tfmodel.p_vector) == length(obj.tfmodel.p_vector), 'TFDefaultEngine:TFDefaultEngine', 'TFDefaultEngine currently only supports KL divergence calculation' );
            obj.kl_divergence = zeros(size(obj.tfmodel.coupling_matrix,1), obj.iteration_number);
                
            obj.gtp_rules = obj.tfmodel.update_rules();
        end

        function [] = factorize(obj)
            for it_num = 1:obj.iteration_number
                obj.tfmodel.factorization_model{1}.data(1:10)
                obj.tfmodel.X_hat_tensors(1).data(1:10)
                iteration_tic = tic;
                display([ char(10) 'iteration ' num2str(it_num) ]);
                for rule_ind = 1:length(obj.gtp_rules)
                    if iscell( obj.gtp_rules{rule_ind}{3} )
                        input = num2str(cellfun( @(x) x.id, obj.gtp_rules{rule_ind}{3} ));
                    else
                        input = obj.gtp_rules{rule_ind}{3};
                    end
                    display_rule( obj.gtp_rules{rule_ind}, rule_ind, 'Executing ' );

                    %for ri = 1:length(obj.gtp_rules)
                    %    display_rule( obj.gtp_rules{ri}, ri, 'ALL RULES ' );
                    %end

                    if obj.gtp_rules{rule_ind}{1} == 'GTP'
                        assert( sum_all_dims(size(obj.gtp_rules{rule_ind}{2}.data)) ~= 0, 'TFDefaultEngine:TFDefaultEngine', 'GTP operation requires output tensor with non-zero data' );
                        input = obj.gtp_rules{rule_ind}{3};
                        gtp( obj.gtp_rules{rule_ind}{2}, input{:} );
                    else
                        obj.gtp_rules{rule_ind}{2}.data = eval( obj.gtp_rules{rule_ind}{3} );
                    end
                end


                for fm_ind = 1:2:length(obj.tfmodel.factorization_model)
                    X_data = obj.tfmodel.factorization_model{fm_ind}.data;
                    v = round(fm_ind/2);
                    X_hat_data = obj.tfmodel.X_hat_tensors(v).data;
                    kl_divergence =  X_data .* log( X_data ) - X_data .* log(  X_hat_data ) - X_data + X_hat_data;
                    for di = 1:sum(obj.tfmodel.coupling_matrix(v,:))
                        kl_divergence = sum(kl_divergence);
                    end
                    obj.kl_divergence( v, it_num ) = kl_divergence;
                end

                display( ['iteration time ' num2str(toc(iteration_tic)) ' seconds divergences ' regexprep(num2str( obj.kl_divergence( :, it_num ) ), '\s*', ',') ] );
            end % end iteration
        end
    end
end