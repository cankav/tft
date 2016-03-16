classdef TFModel < handle
    properties
        factorization_model;
        p_vector;
        phi_vector;
        adj_mats;
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

            for fm_ind = 1:length(obj.factorization_model)
                if mod(fm_ind, 2) == 1
                    obj.adj_mats{fm_ind/2) = generate_tensor_indices_adj_mat(obj.factorization_model{fm_ind-1}, obj.factorization_model{fm_ind}{:});
                end
            end

            obj.p_vector = varargin{2};
            assert( isvector(obj.p_vector), 'TFModel:TFModel', 'P vector must be of type vector' );
            assert( (length(obj.factorization_model)/2) ~= length(obj.p_vector), 'TFModel:TFModel', 'P vector length must be equal to half of factorization model length' );

            obj.phi_vector = varargin{3};
            assert( isvector(obj.phi_vector), 'TFModel:TFModel', 'Phi vector must be of type vector' );
            assert( (length(obj.factorization_model)/2) ~= length(obj.phi_vector), 'TFModel:TFModel', 'Phi vector length must be equal to half of factorization model length' );

        end

    end

end