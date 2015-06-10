function [] = gtp_full(output_tensor, varargin)
    assert( isa(output_tensor, 'Tensor'), 'gtp_full', 'output_tensor must be a Tensor instance' )
    for i = 1:length(varargin)
        assert( isa(varargin{i}, 'Tensor'), 'gtp_full', 'varargin elements should be Tensor insantences' )
    end

    pre_process();

    global tft_indices;

    f_str = 'F = Tensor(';
    for iind = 1:length(tft_indices)
        if iind ~= 1
            f_str = [ f_str ',' ];
        end
        f_str = [ f_str 'tft_indices(' num2str(iind) ')' ];
    end
    f_str = [ f_str ');' ];
    eval(f_str);
    
    % multiply all input tensors' data into F.data
    F.data = ones( tft_indices.cardinality );
    tic;
    for i = 1:length(varargin)
        F.data = bsxfun( @times, F.data, varargin{i}.data );
    end
    toc

    % contract F tensor over indices not present in output_tensor
end