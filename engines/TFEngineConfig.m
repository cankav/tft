classdef TFEngineConfig < handle
    properties
        gtp_rules;
        tfmodel;
        iteration_number;
        timeout=0;
    end

    methods
        function obj = TFEngineConfig(varargin)
            assert( length(varargin) == 2, 'TFEngineConfig:TFEngineConfig', 'TFEngineConfig needs exactly 2 arguments: factorization_model, number of iterations' );
            assert( isa(varargin{1}, 'TFModel'), 'TFEngineConfig:TFEngineConfig', 'First argument must be of an instance of TFModel' );
            assert( isnumeric(varargin{2}) && length(varargin{1})==1, 'TFEngineConfig:TFEngineConfig', 'Iteration number must be a single integer' );

            obj.tfmodel = varargin{1};
            obj.iteration_number = varargin{2};

            assert( sum(obj.tfmodel.p_vector) == length(obj.tfmodel.p_vector), 'TFEngineConfig:TFEngineConfig', 'TFEngineConfig currently only supports KL divergence calculation' );
                
            obj.gtp_rules = obj.tfmodel.update_rules();
        end
    end
end