classdef TFModel < handle
    properties
        factorization_model;
        p_vector;
        phi_vector;

        adj_mats;
        coupling_matrix;
        z_alpha;
        z_alpha_tensor_ids;

        % extra tensor data storage
        x_hat_tensors;
        d1_alpha;
        d2_alpha;
        d1_Q_v;
        d2_Q_v;
    end

    methods

        function obj = TFModel(varargin)
            assert( length(varargin) == 3, 'TFModel:TFModel', 'TFModel needs exactly 3 arguments: factorization_model, p vector, phi vector' );

            obj.factorization_model = varargin{1};
            assert( iscell(obj.factorization_model), 'TFModel:TFModel', 'Factorization model must be of type cell' );
            assert( length(obj.factorization_model) >= 2, 'TFModel:TFModel', 'Factorization model must have at least 2 elements' );
            assert( mod(length(obj.factorization_model),2) == 0 , 'TFModel:TFModel', 'Factorization model must have even number of elements' );

            for fm_ind = 1:length(obj.factorization_model)
                if mod(fm_ind, 2) == 1
                    assert( length(obj.factorization_model{fm_ind}) == 1, 'TFModel:TFModel', 'Odd indices of factorization model must be of length 1' );
                    assert( isa(obj.factorization_model{fm_ind}, Tensor), 'TFModel:TFModel', 'All odd indexed elements of factorization model must be of type Tensor' );
                else
                    assert( length(obj.factorization_model{fm_ind}) >= 2, 'TFModel:TFModel', 'Even indices of factorization model must be of length >= 2' );
                    assert( iscell(obj.factorization_model{fm_ind}), 'TFModel:TFModel', 'Even indices of factorization model must be of type cell' );
                    
                    for factor_ind = 1:length(obj.factorization_model{fm_ind})
                        assert( isa(obj.factorization_model{fm_ind}{factor_ind}, Tensor), 'TFModel:TFModel', 'Elements of all even indexed elements of  factorization model must be of type Tensor' );
                    end
                end
            end

            for fm_ind = 1:2:length(obj.factorization_model)
                tmp = obj.factorization_model{fm_ind+1};
                obj.adj_mats{end+1) = generate_tensor_indices_adj_mat(obj.factorization_model{fm_ind}, tmp{:});
            end

            obj.p_vector = varargin{2};
            assert( isvector(obj.p_vector), 'TFModel:TFModel', 'P vector must be of type vector' );
            assert( (length(obj.factorization_model)/2) ~= length(obj.p_vector), 'TFModel:TFModel', 'P vector length must be equal to half of factorization model length' );

            obj.phi_vector = varargin{3};
            assert( isvector(obj.phi_vector), 'TFModel:TFModel', 'Phi vector must be of type vector' );
            assert( (length(obj.factorization_model)/2) ~= length(obj.phi_vector), 'TFModel:TFModel', 'Phi vector length must be equal to half of factorization model length' );

            % generate x_hat tensors
            obj.x_hat_tensors = [];
            for fm_ind = 1:2:length(obj.factorization_model)
                indices = obj.factorization_model{fm_ind}.indices;
                x_hat_tensors(end+1) = Tensor(indices{:});
                %x_hat_tensors(end).data = rand(cellfun( @(index) index.cardinality, indices ));
            end

            % generate z_alpha
            obj.z_alpha_tensor_ids = [];
            obj.z_alpha = [];
            for fm_ind = 2:2:length(obj.factorization_model)
                for factor_ind = 1:length(obj.factorization_model{fm_ind})
                    if sum(obj.z_alpha_tensor_ids == obj.factorization_model{fm_ind}{factor_ind}.id) == 0
                        obj.z_alpha_tensor_ids(end+1) = obj.factorization_model{fm_ind}{factor_ind}.id;
                        obj.z_alpha(end+1) = obj.factorization_model{fm_ind}{factor_ind};
                    end
                end
            end

            % generate coupling tensor
            obj.coupling_matrix = sparse();
            for fm_ind = 2:2:length(obj.factorization_model)
                for factor_ind = 1:length(obj.factorization_model{fm_ind})
                    obj.coupling_matrix(fm_ind/2, obj.factorization_model{fm_ind}{factor_ind}.id) = 1;
                end
            end
        end

        function gtp_rules = update_rules(obj)
            gtp_rules = {};
            % gtp rule: { '<operation_type>', <output_tensor>, <input> }
            % <operation_type> may be a string descibing a matlab operation or the letters GTP for gtp operation
            % <output_tensor> is a Tensor object describing output tensor
            % <input> is an array of Tensor objects or a string which will be executed and results will be stored in output_tensor

            % prepare temporary tensors required for gctf operation
            obj.d1_alpha = [];
            obj.d2_alpha = [];
            for zat_ind = 1:length(obj.z_alpha)
                indices = obj.z_alpha(zat_ind).indices;
                obj.d1_alpha(end+1) = Tensor(indices{:});
                obj.d2_alpha(end+1) = Tensor(indices{:});
            end
            obj.d1_Q_v = [];
            obj.d2_Q_v = [];
            for v = find( obj.coupling_matrix(:, obj.z_alpha_tensor_ids(zat_ind)) )
                indices = obj.x_hat_tensors(v).indices;
                obj.d1_Q_v(end+1) = Tensor(indices{:});
                obj.d2_Q_v(end+1) = Tensor(indices{:});
            end

            % update each x_hat
            for x_hat_ind = 1:length(obj.x_hat_tensors)
                tmp = obj.factorization_model{x_hat_ind*2};
                gtp_rules.append( { obj.x_hat_tensors{x_hat_ind}, tmp{:} } );
            end

            % update each Z_alpha
            for zat_ind = 1:length(obj.z_alpha)
                for v = find( obj.coupling_matrix(:, obj.z_alpha_tensor_ids(zat_ind)) )
                    gtp_rules.append( {'=', 'd1', 'obj.z_alpha(' num2sstr(zat_ind) ')' }
                    gtp_rules.append( {'=', 'd2', 'obj.z_alpha(' num2str(zat_ind) ')' }

                    gtp_rules.append( {'=', 'd1_Q', 'obj.factorization_model{' num2str(v*2) '} .* spfun(@(x) x.^-' num2str(obj.p_vector(v)) ', obj.x_hat_tensors{' num2str(v) '});' ];

                    'contract2' 
                    tmpmodel.delta( tmpmodel.get_factor_index(obj.unique_latent_factors(char(ulfk(alpha)))),...
                                    d1_x_name, [], settings, d1_x_A);

                    if first_v
                        eval([ d1_name ' = obj.phi(v)^-1 *' d1_x_name ';']);
                    else
                        eval([ d1_name ' = ' d1_name ' + obj.phi(v)^-1 * ' d1_x_name ';']);
                    end


                    d2_x_A = TFFactor('name', ...
                                    ['D2_Z' num2str(alpha) '_X' num2str(v) '_A'], ...
                                    'type', 'latent', ...
                                    'dims', obj.observed_factors(v).dims, ...
                                    'isSparse', obj.isSparse);
                    d2_x_A_name = d2_x_A.get_data_name();
                    eval([ 'global ' d2_x_A_name ]);
                    eval( [ d2_x_A_name ' = spfun(@(x) x.^(1-' num2str(obj.p(v)) '), ' hat_X_data_name  ');' ]);

                    tmpmodel = obj.pltf_models(v);

                    'contract3'
                    
                    tmpmodel.delta( tmpmodel.get_factor_index(obj.unique_latent_factors(char(ulfk(alpha)))), ...
                                    d2_x_name, [], settings, d2_x_A);

                    if first_v
                        eval([ d2_name ' = obj.phi(v)^-1 *' d2_x_name ';']);
                    else
                        eval([ d2_name ' = ' d2_name ' + obj.phi(v)^-1 *' d2_x_name ';']);
                    end

                    first_v = false;
                end % v = 1:length(obj.observed_factors)

                % update Z_alpha with d1/d2
                eval([ Z_name '((' Z_name ' ~=0)) = nonzeros(' Z_name ') .* nonzeros(' d1_name ') ./ ' ...
                       'nonzeros(' d2_name ');' ]);
            end % alpha = 1:length(ulfk)




            if strcmp( settings.operation_mode, 'compute' )
                % calculate KL divergence
                kls = zeros(1, length(obj.observed_factors));
                for v = 1:length(obj.observed_factors)
                    hat_X_data_name = hat_X_v(v).get_data_name();
                    X_name = obj.observed_factors(v).get_data_name();

%                    newmodel = obj.pltf_models(v);
%                    'contract4'
%                    %display(['hat_X_data_name: ' hat_X_data_name]);
%                    [ ~ ] = ...
%                        newmodel.contract_all( settings, hat_X_data_name, ...
%                                               newmodel.factor_ind_masks(newmodel.observed_factor_index(), :), ...
%                                               '', ...
%                                               newmodel.observed_factor().get_data_name());

                    if obj.isSparse

                        tcmd = [ 'kl = nonzeros( ' X_name ' ) .* log( nonzeros(' X_name ') ) -' ...
                                 'nonzeros(' X_name ') .* log( nonzeros(' hat_X_data_name ') ) -' ...
                                 'nonzeros(' X_name ') + nonzeros(' hat_X_data_name ') ;' ];

                    else
                        tcmd = [ 'kl = ' X_name '.* log( ' X_name ' ) -' ...
                                 X_name ' .* log( ' hat_X_data_name ' ) -' ...
                                 X_name ' + ' hat_X_data_name ' ;' ];

                    end
                    eval( tcmd );

%                    eval ( [ 't = (' hat_X_data_name ' .* ' X_name ') .* ' ...
%                             ' (log( (' hat_X_data_name ' .* ' X_name ') ) - ' ...
%                             'log(' X_name ...
%                             ') ) - ( ' hat_X_data_name ' .* ' X_name ')' ...
%                             '+ ' X_name ...
%                             ';' ]);


                    for di = 1:length(obj.observed_factors(v).dims)
                        kl = sum(kl);
                    end
                    kls(v) = kl;
                end
            else
                kls = 0;
            end

        end

    end

end