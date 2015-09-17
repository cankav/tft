function tensor = create_tensor( index_ids, init_type )
    global tft_indices

    index_ids = unique( index_ids );
    new_tensor_indices = [];
    data_cardinalities = ones( length(tft_indices), 1 );
    for tftii = 1:length(tft_indices)
        data_cardinalities(tftii) = tft_indices(tftii).cardinality;
        index_count = sum( index_ids == tft_indices(tftii).id );
        if index_count == 1
            new_tensor_indices = [new_tensor_indices tft_indices(tftii)];
        elseif index_count ~= 0
            throw( MException( 'create_tensor', 'Tensor_indices must contain no or one copy of an index.' ) );
        end
    end

    if length(new_tensor_indices) == 0
        throw( MException( 'create_tensor:create_tensor', 'Refusing to create tensor with empty indices' ) );
    end

    tensor = Tensor( new_tensor_indices );
    tensor.tft_indices = tft_indices;
    if strcmp( init_type, 'zeros' )
        tensor.data = zeros( data_cardinalities' );
    elseif strcmp( init_type, 'ones' )
        tensor.data = ones( data_cardinalities' );
    else
        tensor.data = rand( data_cardinalities' );
    end
end