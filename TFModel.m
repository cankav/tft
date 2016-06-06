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

            % generate coupling matrix
            obj.coupling_matrix = sparse(length(obj.factorization_model)/2, length(obj.factorization_model{fm_ind}));
            for fm_ind = 2:2:length(obj.factorization_model)
                for factor_ind = 1:length(obj.factorization_model{fm_ind})
                    obj.coupling_matrix(fm_ind/2, obj.factorization_model{fm_ind}{factor_ind}.id) = 1;
                end
            end
        end

        function [] = display_update_rules(obj)
            update_rules = obj.update_rules();
            for rule_ind = 1:length(update_rules)
                display_rule( update_rules{rule_ind}, rule_ind );
            end
        end

        function gtp_rules = update_rules(obj)
            gtp_rules = {};
            % gtp rule: { '<operation_type>', <output_tensor>, <input> }
            % <operation_type> may be a string descibing a matlab operation or the letters GTP for gtp operation
            % <output_tensor> is a Tensor object describing output tensor
            % <input> is a cell of Tensor objects or a string which will be executed and results will be stored in output_tensor

            pre_process();

            % prepare X_hat tensors
            obj.X_hat_tensors = [];
            xhat_index = 1;
            for fm_ind = 1:2:length(obj.factorization_model)
                obj.X_hat_tensors = [ obj.X_hat_tensors create_tensor( cellfun( @(index) index.id, obj.factorization_model{fm_ind}.indices ), 'random' ) ];
                obj.X_hat_tensors(end).name = ['Xhat_' num2str(xhat_index)];
                xhat_index = xhat_index + 1;
            end

            % prepare temporary tensors required for gctf operation
            obj.d1_alpha = [];
            obj.d2_alpha = [];
            for alpha = 1:length(obj.Z_alpha)
                zalpha_indices = obj.Z_alpha(alpha).indices;
                obj.d1_alpha = [ obj.d1_alpha create_tensor( cellfun( @(index) index.id, zalpha_indices ), 'zeros' ) ];
                obj.d1_alpha(end).name = ['d1_alpha_' num2str(alpha)];
                obj.d2_alpha = [ obj.d2_alpha create_tensor( cellfun( @(index) index.id, zalpha_indices ), 'zeros' ) ];
                obj.d2_alpha(end).name = ['d2_alpha_' num2str(alpha)];
                obj.d1_delta = [ obj.d1_delta create_tensor( cellfun( @(index) index.id, zalpha_indices ), 'zeros' ) ];
                obj.d1_delta(end).name = ['d1_delta_' num2str(alpha)];
                obj.d2_delta = [ obj.d2_delta create_tensor( cellfun( @(index) index.id, zalpha_indices ), 'zeros' ) ];
                obj.d2_delta(end).name = ['d2_delta_' num2str(alpha)];
            end
            obj.d1_Q_v = [];
            obj.d2_Q_v = [];
            for v = 1:length(obj.X_hat_tensors) %find( obj.coupling_matrix(:, obj.Z_alpha_tensor_ids(alpha)) )
                X_indices = obj.X_hat_tensors(v).indices;
                obj.d1_Q_v = [ obj.d1_Q_v create_tensor( cellfun( @(index) index.id, X_indices ), 'zeros' ) ];
                obj.d1_Q_v(end).name = ['d1_Q_v_' num2str(v)];
                
                obj.d2_Q_v = [ obj.d2_Q_v create_tensor( cellfun( @(index) index.id, X_indices ), 'zeros' ) ];
                obj.d2_Q_v(end).name = ['d2_Q_v_' num2str(v)];
            end

            % update each Z_alpha
            for alpha = 1:length(obj.Z_alpha)
                % update each X_hat
                for X_hat_ind = 1:length(obj.X_hat_tensors)
                    latent_tensors = obj.factorization_model{X_hat_ind*2};
                    gtp_rules{end+1} = { 'GTP', obj.X_hat_tensors(X_hat_ind), latent_tensors };
                end

                first_v = true;
                v_indices = find( obj.coupling_matrix(:, obj.Z_alpha_tensor_ids(alpha)) );
                for v_ii = length(v_indices)
                    v_index = v_indices(v_ii);
                    observed_tensor_fm_ind = v_index*2-1;
                    gtp_rules{end+1} = { '=',
                                        obj.d1_Q_v(v_index),
                                        ['obj.config.tfmodel.factorization_model{' num2str(observed_tensor_fm_ind) '}.data .* arrayfun(@(x) x.^-' num2str(obj.p_vector(v_index)) ', obj.config.tfmodel.X_hat_tensors(' num2str(v_index) ').data);'] };

                    Z_alpha_inds = find(obj.coupling_matrix(v_index,:));
                    Z_alpha_bar_inds = Z_alpha_inds;
                    zalpha = obj.Z_alpha(alpha);
                    Z_alpha_bar_inds( Z_alpha_bar_inds == zalpha.id ) = [];
                    Z_alpha_bar_tensors = num2cell(obj.Z_alpha( arrayfun( @(x) (sum(Z_alpha_bar_inds==x.id)==1), obj.Z_alpha )));
                    %display(Z_alpha_bar_inds);
                    gtp_rules{end+1} = { 'GTP',
                                        obj.d1_delta(alpha),
                                        {obj.d1_Q_v(v_index),  Z_alpha_bar_tensors{:} } };

                    if first_v
                        gtp_rules{end+1} = { '=', obj.d1_alpha(alpha), ['obj.config.tfmodel.phi_vector(' num2str(v_index) ')^-1 .* obj.config.tfmodel.d1_delta(' num2str(alpha) ').data'] };
                    else
                        gtp_rules{end+1} = { '=', obj.d1_alpha(alpha), ['obj.config.tfmodel.d1_alpha(' num2str(alpha) ').data + obj.config.tfmodel.phi_vector(' num2str(v_index) ')^-1 .* obj.config.tfmodel.d1_delta(' num2str(alpha) ').data'] };
                    end

                    gtp_rules{end+1} = { '=',
                                        obj.d2_Q_v(v_index),
                                        ['arrayfun(@(x) x.^(1-' num2str(obj.p_vector(v_index)) '), obj.config.tfmodel.X_hat_tensors(' num2str(v_index) ').data);'] };

                    Z_alpha_inds = find(obj.coupling_matrix(v_index,:));
                    Z_alpha_bar_inds = Z_alpha_inds;
                    zalpha = obj.Z_alpha(alpha);
                    Z_alpha_bar_inds( Z_alpha_bar_inds == zalpha.id ) = [];
                    Z_alpha_bar_tensors = num2cell(obj.Z_alpha( arrayfun( @(x) (sum(Z_alpha_bar_inds==x.id)==1), obj.Z_alpha ) ));
                    gtp_rules{end+1} = { 'GTP',
                                        obj.d2_delta(alpha),
                                        {obj.d2_Q_v(v_index), Z_alpha_bar_tensors{:}} };

                    if first_v
                        gtp_rules{end+1} = { '=', obj.d2_alpha(alpha), ['obj.config.tfmodel.phi_vector(' num2str(v_index) ')^-1 .* obj.config.tfmodel.d2_delta(' num2str(alpha) ').data'] };
                        first_v = false;
                    else
                        gtp_rules{end+1} = { '=', obj.d2_alpha(alpha), ['obj.config.tfmodel.d2_alpha(' num2str(alpha) ').data + obj.config.tfmodel.phi_vector(' num2str(v_index) ')^-1 .* obj.config.tfmodel.d2_delta(' num2str(alpha) ').data'] };
                    end

                    first_v = false;
                end % v_index loop

                % update Z_alpha with d1/d2
                gtp_rules{end+1} = { '=', obj.Z_alpha(alpha), ['obj.config.tfmodel.Z_alpha(' num2str(alpha) ').data .* obj.config.tfmodel.d1_alpha(' num2str(alpha) ').data ./ obj.config.tfmodel.d2_alpha('  num2str(alpha) ').data'] };
            end % alpha loop

        end % update_rules

    end % methods

end