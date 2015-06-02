classdef Tensor < handle

    properties
        data = [];
        indices = {};
        reshaped = false;
    end

    methods

        function obj = Tensor(varargin)
            for i = 1:length(varargin)
                assert( isa(varargin{i}, 'Index'), 'Tensor:Tensor', 'Tensor constructor arguments must be of type Index' )
                obj.indices{end+1} = varargin{i};
            end
        end

        function sref = subsref(obj,s)
            switch s(1).type
                case '.'
                  sref = builtin('subsref',obj,s);
                case '()'
                  display(s);
                  %sref = 
                case '{}'
                  sref = builtin('subsref',obj,s);
            end
        end

    end

end