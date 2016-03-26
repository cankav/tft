classdef TFModel < handle
    properties
        factorization_model;
        p_vector;
        phi_vector;

        adj_mats;
        coupling_matrix;
        Z_alpha;
        Z_alpha_tensor_ids;

        % extra tensor data storage
        X_hat_tensors;
        d1_alpha;
        d2_alpha;
        d1_delta;
        d2_delta;
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
                    assert( isa(obj.factorization_model{fm_ind}, 'Tensor'), 'TFModel:TFModel', 'All odd indexed elements of factorization model must be of type Tensor' );
                else
                    assert( length(obj.factorization_model{fm_ind}) >= 2, 'TFModel:TFModel', 'Even indices of factorization model must be of length >= 2' );
                    assert( iscell(obj.factorization_model{fm_ind}), 'TFModel:TFModel', 'Even indices of factorization model must be of type cell' );
                    
                    for factor_ind = 1:length(obj.factorization_model{fm_ind})
                        assert( isa(obj.factorization_model{fm_ind}{factor_ind}, 'Tensor'), 'TFModel:TFModel', 'Elements of all even indexed elements of  factorization model must be of type Tensor' );
                    end
                end
            end

            for fm_ind = 1:2:length(obj.factorization_model)
                tmp = obj.factorization_model{fm_ind+1};
                obj.adj_mats{end+1} = generate_tensor_indices_adj_mat(obj.factorization_model{fm_ind}, tmp{:});
            end

            obj.p_vector = varargin{2};
            assert( isvector(obj.p_vector), 'TFModel:TFModel', 'P vector must be of type vector' );
            assert( (length(obj.factorization_model)/2) == length(obj.p_vector), 'TFModel:TFModel', ['p vector length must be equal to half of factorization model length. p vector length ' num2str(length(obj.p_vector)) ' factorization model length ' num2str(length(obj.factorization_model))] );

            obj.phi_vector = varargin{3};
            assert( isvector(obj.phi_vector), 'TFModel:TFModel', 'Phi vector must be of type vector' );
            assert( (length(obj.factorization_model)/2) == length(obj.phi_vector), 'TFModel:TFModel', 'phi vector length must be equal to half of factorization model length' );

            % generate X_hat tensors
            obj.X_hat_tensors = [];
            for fm_ind = 1:2:length(obj.factorization_model)
                indices = obj.factorization_model{fm_ind}.indices;
                obj.X_hat_tensors = [ obj.X_hat_tensors Tensor(indices{:}) ];
                %X_hat_tensors(end).data = rand(cellfun( @(index) index.cardinality, indices ));
            end

            % generate Z_alpha
            obj.Z_alpha_tensor_ids = [];
            obj.Z_alpha = [];
            for fm_ind = 2:2:length(obj.factorization_model)
                for factor_ind = 1:length(obj.factorization_model{fm_ind})
                    if sum(obj.Z_alpha_tensor_ids == obj.factorization_model{fm_ind}{factor_ind}.id) == 0
                        obj.Z_alpha_tensor_ids(end+1) = obj.factorization_model{fm_ind}{factor_ind}.id;
                        obj.Z_alpha = [ obj.Z_alpha obj.factorization_model{fm_ind}{factor_ind} ];
                    end
                end
            end

            % generate coupling tensor
            obj.coupling_matrix = sparse(length(obj.factorization_model)/2, length(obj.factorization_model{fm_ind}));
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
            % <input> is an cell of Tensor objects or a string which will be executed and results will be stored in output_tensor

            % prepare temporary tensors required for gctf operation
            obj.d1_alpha = [];
            obj.d2_alpha = [];
            for alpha = 1:length(obj.Z_alpha)
                zalpha = obj.Z_alpha(alpha);
                indices = zalpha.indices;
                obj.d1_alpha = [obj.d1_alpha Tensor(indices{:})];
                obj.d2_alpha = [obj.d2_alpha Tensor(indices{:})];
                obj.d1_delta = [obj.d1_delta Tensor(indices{:})];
                obj.d2_delta = [obj.d2_delta Tensor(indices{:})];
            end
            obj.d1_Q_v = [];
            obj.d2_Q_v = [];
            for v = find( obj.coupling_matrix(:, obj.Z_alpha_tensor_ids(alpha)) )
                xhat = obj.X_hat_tensors(v);
                indices = xhat.indices;
                obj.d1_Q_v = [obj.d1_Q_v Tensor(indices{:})];
                obj.d2_Q_v = [obj.d2_Q_v Tensor(indices{:})];
            end

            % update each X_hat
            for X_hat_ind = 1:length(obj.X_hat_tensors)
                latent_tensors = obj.factorization_model{X_hat_ind*2};
                gtp_rules{end+1} = { 'GTP', obj.X_hat_tensors(X_hat_ind), latent_tensors };
            end

            % update each Z_alpha
            for alpha = 1:length(obj.Z_alpha)
                first_v = true;
                for v = find( obj.coupling_matrix(:, obj.Z_alpha_tensor_ids(alpha)) )
                    gtp_rules{end+1} = { '=',
                                        obj.d1_Q_v(v),
                                        ['tfmodel.factorization_model{' num2str(v*2) '}.data .* spfun(@(x) x.^-' num2str(obj.p_vector(v)) ', tfmodel.X_hat_tensors(' num2str(v) ').data);'] };

                    Z_alpha_inds = find(obj.coupling_matrix(v,:));
                    Z_alpha_bar_inds = Z_alpha_inds;
                    zalpha = obj.Z_alpha(alpha);
                    Z_alpha_bar_inds( Z_alpha_bar_inds == zalpha.id ) = [];
                    gtp_rules{end+1} = { 'GTP',
                                        obj.d1_delta(alpha),
                                        {obj.d1_Q_v(v), arrayfun( @(x) (sum(Z_alpha_bar_inds==x.id)==1), obj.Z_alpha )} };

                    if first_v
                        gtp_rules{end+1} = { '=', obj.d1_alpha(alpha), ['tfmodel.phi(v)^-1 * tfmodel.delta(' num2str(alpha) ')'] };
                    else
                        gtp_rules{end+1} = { '=', obj.d1_alpha(alpha), ['tfmodel.d1_alpha(' num2str(alpha) ') + tfmodel.phi(v)^-1 * tfmodel.delta(' num2str(alpha) ')'] };
                    end

                    gtp_rules{end+1} = { '=',
                                        obj.d2_Q_v(v),
                                        ['tfmodel.factorization_model{' num2str(v*2) '}.data .* spfun(@(x) x.^(1-' num2str(obj.p_vector(v)) '), tfmodel.X_hat_tensors(' num2str(v) ').data);'] };

                    Z_alpha_inds = find(obj.coupling_matrix(v,:));
                    Z_alpha_bar_inds = Z_alpha_inds;
                    zalpha = obj.Z_alpha(alpha);
                    Z_alpha_bar_inds( Z_alpha_bar_inds == zalpha.id ) = [];
                    gtp_rules{end+1} = { 'GTP',
                                        obj.d2_delta(alpha),
                                        {obj.d2_Q_v(v), arrayfun( @(x) (sum(Z_alpha_bar_inds==x.id)==1), obj.Z_alpha )} };

                    if first_v
                        gtp_rules{end+1} = { '=', obj.d2_alpha(alpha), ['tfmodel.phi(v)^-2 * tfmodel.delta(' num2str(alpha) ')'] };
                        first_v = false;
                    else
                        gtp_rules{end+1} = { '=', obj.d2_alpha(alpha), ['tfmodel.d2_alpha(' num2str(alpha) ') + tfmodel.phi(v)^-1 * tfmodel.delta(' num2str(alpha) ')'] };
                    end

                    first_v = false;
                end % v loop

                % update Z_alpha with d1/d2
                gtp_rules{end+1} = { '=', obj.Z_alpha(alpha), ['tfmodel.d1_alpha(' num2str(alpha) ') ./ tfmodel.d2_alpha('  num2str(alpha) ')'] };
            end % alpha loop

        end

    end

end