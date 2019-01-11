classdef Index < handle

    properties
        id
        cardinality;
        name;
    end

    methods

        function obj = Index(cardinality)
            if nargin ~= 1
                error('Index objects must have a single argument indicating cardinality of the index');
            end

            if ~(isnumeric(cardinality)) || ~(isscalar(cardinality)) || cardinality <= 0
                error('Index cardinality argument must be a positive integer')
            end
            
            global TFT_Index_index
            if length(TFT_Index_index) == 0
                TFT_Index_index = 1;
            else
                TFT_Index_index = TFT_Index_index + 1;
            end
            obj.id = TFT_Index_index;

            obj.cardinality = double(cardinality);

        end

    end

end