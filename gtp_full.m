function [] = gtp_full(output_tensor, varargin)
    assert( isa(output_tensor, 'Tensor'), 'gtp_full', 'output_tensor must be a Tensor instance' )
    for i = 1:length(varargin)
        assert( isa(varargin{i}, 'Tensor'), 'gtp_full', 'varargin elements should be Tensor insantences' )
    end

    pre_process();
    
    %output_tensor.data = bsxfun();

    %bsxfun
end