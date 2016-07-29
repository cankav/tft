function tensor = create_tensor( index_ids, init_type )
    global tft_indices

    index_ids = unique( index_ids );
    new_tensor_indices = {};
    data_cardinalities = ones( length(tft_indices), 1 );
    for ii_ind = 1:length(index_ids)
        new_tensor_indices{end+1} = tft_indices(index_ids(ii_ind));
        data_cardinalities(index_ids(ii_ind)) = tft_indices(index_ids(ii_ind)).cardinality;
    end

    if length(index_ids) == 0
        throw( MException( 'create_tensor:create_tensor', 'Refusing to create tensor with empty indices' ) );
    end

    %tensor_indices = tft_indices(index_ids);
    %data_cardinalities(index_ids) = tft_indices(index_ids).cardinality;
    %tensor = Tensor( tensor_indices(:) );
    tensor = Tensor( new_tensor_indices{:} );
    tensor.tft_indices = tft_indices;
    tensor.index_ids = index_ids;

    global TFT_Tensors
    TFT_Tensors{tensor.id} = tensor;

    if nargin == 2
        if strcmp( class(init_type), 'Tensor' )
            input_tensor = init_type;

            assert( issparse(input_tensor.data), 'create_tensor:create_tensor', 'It is meaningless to use a tensor as init_type if given tensor data is not sparse, something must be wrong' );

            % check indices of tensor, they must be identical to the created tensor
            for ttind = 1:length(tensor.indices)
                if input_tensor.reshaped == 1
                    tensor_indices_ind = input_tensor.original_indices_permute_array(ttind);
                else
                    tensor_indices_ind = ttind;
                end

                assert( tensor.indices{ttind}.id == input_tensor.indices{tensor_indices_ind}.id, 'create_tensor:create_tensor', 'If init_type is given as tensor to determine sparse init indices, indices of given tensor must match with created tensor''s indices one to one' );
            end

            [x_ind, y_ind] = find(input_tensor.data);
            tensor.data = sparse(x_ind, y_ind, rand(nnz(input_tensor.data),1));

            % this init, creates sparse vector of size max(x_ind), but framework needs the vector size to be Index * Index
            % if sparse input_tensor does not contain element in largest index, following code will set tensor.data to correct size
            input_tensor_size = size(input_tensor.data,1);
            if input_tensor_size ~= size(tensor.data,1)
                tensor.data(input_tensor_size,1) = 0;
            end

            'create size'
            size(tensor.data)
            'x_ind size'
            size(x_ind)
            'input size'
            size(input_tensor.data)
            'find size'
            size(find(input_tensor.data))
            'rand size'
            nnz(input_tensor.data)

        else
            if strcmp( init_type, 'zeros' )
                tensor.data = zeros( data_cardinalities' );
            elseif strcmp( init_type, 'ones' )
                tensor.data = ones( data_cardinalities' );
            elseif strcmp( init_type, 'random' )
                tensor.data = rand( data_cardinalities' );
            else
                throw( MException( 'create_tensor:create_tensor', 'Unknown init_type' ) );
            end
        end
    end
end