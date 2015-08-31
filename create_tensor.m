function tensor = create_tensor( tensor_indices )
    global tft_indices

    tensor_indices = unique( tensor_indices );
    indices = []
    data_cardinalities = ones( length(tft_indices) );    
    for tftii = 1:length(tft_indices)
        if sum( tensor_indices == tft_indices(tftii) ) == 1
            indices = [indices tft_indices(tfii)];
            data_cardinalities = tft_indices(tftii).cardinality;
        elseif sum( tensor_indices == tft_indices(tftii) ) == 0
            data_cardinalities = 1;
        else
            throw( MException( 'create_tensor', 'tensor_indices must contain a single copy of an index.' ) );
        end
    end

    if length(indices) == 0
        throw( MException( 'create_tensor', 'can not create tensor with empty indices' ) );
    end

    tensor = Tensor( indices );
    tensor.tft_indices = tft_indices;
    tensor.data = rand( data_cardinalities );
end