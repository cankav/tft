function [] = gtp(output_tensor, varargin)
    assert( isa(output_tensor, 'Tensor'), 'gtp', 'output_tensor must be a Tensor instance' )
    for i = 1:length(varargin)
        assert( isa(varargin{i}, 'Tensor'), 'gtp', 'varargin elements should be Tensor insantences' )
    end

    assert( length(varargin) > 1, 'gtp', 'number of input tensors must be greater than one' )

    tic;
    pre_process();
    display( [ ' pre_process time: ' num2str(toc) ] );

    global tft_indices;

    % generate adjacency matrix for gtp operation
    % adj_mat( tensor_id, tft_indices_id ) is equal to 1 if tensor with tensor_id has data on dimension tft_indices_id
    % TODO: adj_mat should be built once by higher level functions and passed to gtp as input
    global TFT_Tensor_index
    adj_mat = generate_tensor_indices_adj_mat([varargin output_tensor]);

    global TFT_Tensor_index
    global TFT_Tensors
    contraction_indices = tft_indices ( logical ( sum( bsxfun(
                              @times,
                              adj_mat(output_tensor.id, : ) == 0,
                              adj_mat( find(1:TFT_Tensor_index ~= output_tensor.id), : ) == 1,
                              ) ) ) )

    % clear output_tensor data to save memory, tmp_tensor.data will be set as output_tensor.data
    output_tensor.data = [];

    for ci_ind = 1:length(contraction_indices)

        % tensors with data on current contraction dimension
        ci_contraction_tensor_ids = find(adj_mat( :, contraction_indices(ci_ind.id) ) == 1);

        if length( ci_contraction_tensor_ids ) == 1
            % no \prod operation, copy orig data as temporary data, do not modify input data
            tmp_tensor = create_tensor( [ TFT_Tensors{ci_contraction_tensor_ids(1)}.indices ] );
            tmp_tensor.data = tmp_tensor.data;
            adj_mat(tmp_tensor.id, tmp_tensor.index_ids) = 1;
            % \sum operation
            tmp_tensor.data = sum( tmp_tensor.data, contraction_indices(ci_ind) );
            % is this case meaningful? no interaction with other indices means this operation can be skipped?

        else
            % init temporary tensor for current contraction dimension
            tmp_tensor_index_ids = [];
            for ctid_ind = 1:length(ci_contraction_tensor_ids)
                tmp_tensor_index_ids = [tmp_tensor_index_ids TFT_Tensors{ci_contraction_tensor_ids(ctid_ind)}.index_ids];
            end
            tmp_tensor = create_tensor( tmp_tensor_index_ids, 'ones' );

            % multiply contraction tensors into temporary tensor
            for ctid_ind = 1:length(ci_contraction_tensor_ids)
                tmp_tensor.data = bsxfun( @times, tmp_tensor.data, TFT_Tensors{ci_contraction_tensor_ids(ctid_ind)}.data );

                adj_mat(tmp_tensor.id, tmp_tensor.index_ids) = 1;
                ci_contraction_tensor_ids(end+1) = tmp_tensor.id;

                ci_contraction_tensor_ids = ci_contraction_tensor_ids(3:end);
            end

            % \sum operation
            tmp_tensor.data = sum( tmp_tensor.data, contraction_index_inds );
        end

    end
end