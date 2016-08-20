function [] = gtp(output_tensor, varargin)
    assert( isa(output_tensor, 'Tensor'), 'gtp', 'output_tensor must be a Tensor instance' )
    for i = 1:length(varargin)
        assert( isa(varargin{i}, 'Tensor'), 'gtp', 'varargin elements should be Tensor insantences' )
    end

    assert( length(varargin) > 1, 'gtp', 'number of input tensors must be greater than one' )

    pre_process();

    % generate adjacency matrix for gtp operation
    % TODO: adj_mat should be built once by higher level functions and passed to gtp as input
    %       to save time in case of recurrent gtp operations
    args = {output_tensor varargin{:}};
    adj_mat = generate_tensor_indices_adj_mat( args{:} );

    global tft_indices;
    global TFT_Tensor_index;
    global TFT_Tensors;
    contraction_indices = tft_indices ( logical ( sum( bsxfun( @times, ...
                                                      adj_mat(output_tensor.id, : ) == 0, ...
                                                      adj_mat( find(1:(TFT_Tensor_index-1) ~= output_tensor.id), : ) == 1 ...
                                                      ) ) ) );

    used_tensor_ids = [];
    for ci_ind = 1:length(contraction_indices)
        % tensors with data on current contraction dimension
        ci_contraction_tensor_ids = find(adj_mat( :, contraction_indices(ci_ind).id ) == 1);

        % filter tensors which have been used before in this GTP operation
        ci_contraction_tensor_ids( ismember(ci_contraction_tensor_ids, used_tensor_ids) ) = [];
        used_tensor_ids = [used_tensor_ids ci_contraction_tensor_ids'];

        % init temporary tensor for current contraction dimension
        tmp_tensor_index_ids = [];
        for ctid_ind = 1:length(ci_contraction_tensor_ids)
            tmp_tensor_index_ids = [tmp_tensor_index_ids TFT_Tensors{ci_contraction_tensor_ids(ctid_ind)}.index_ids];
        end

        tmp_tensor_index_ids = unique(tmp_tensor_index_ids);
        % do not add contraction dimension to tensor's dimensions
        tmp_tensor_index_ids( tmp_tensor_index_ids==contraction_indices(ci_ind).id ) = [];
        
        tmp_tensor = create_tensor( tmp_tensor_index_ids, 'ones' );
        tmp_tensor.name = 'gtp_tmp_tensor';
        adj_mat(tmp_tensor.id, tmp_tensor.index_ids) = 1;

        % \prod operation: multiply contraction tensors into temporary tensor
        for ctid_ind = 1:length(ci_contraction_tensor_ids)
            tmp_tensor.data = bsxfun( @times, tmp_tensor.data, TFT_Tensors{ci_contraction_tensor_ids(ctid_ind)}.data );
        end

        % \sum operation: contract temporary tensor
        tmp_tensor.data = sum( tmp_tensor.data, contraction_indices(ci_ind).id );
    end

    % last temporary tensor data holds the result, 
    % expecting output_tensor to be defined beforehand without the data field
    output_tensor.data = tmp_tensor.data;
    %size(output_tensor.data)
end