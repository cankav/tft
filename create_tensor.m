function tensor = create_tensor( index_ids, init_type )
    global tft_indices

    index_ids = unique( index_ids );
    new_tensor_indices = {};
    data_cardinalities = ones( length(tft_indices), 1 );
    for tftii = 1:length(tft_indices)
        index_count = sum( index_ids == tft_indices(tftii).id );
        if index_count == 1
            new_tensor_indices{end+1} = tft_indices(tftii);
            data_cardinalities(tftii) = tft_indices(tftii).cardinality;
        elseif index_count ~= 0
            throw( MException( 'create_tensor', 'Tensor_indices must contain no or one copy of an index.' ) );
        end
    end

    if length(new_tensor_indices) == 0
        throw( MException( 'create_tensor:create_tensor', 'Refusing to create tensor with empty indices' ) );
    end

    tensor = Tensor( new_tensor_indices{:} );
    tensor.tft_indices = tft_indices;
    tensor.index_ids = index_ids;

    global TFT_Tensors
    TFT_Tensors{tensor.id} = tensor;

    if nargin == 2
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