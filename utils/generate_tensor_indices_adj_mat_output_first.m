function [adj_mat] = generate_tensor_indices_adj_mat_output_first(varargin)
    % adj_mat( tensor_id, tft_indices_id ) is equal to 1 if tensor with tensor_id has data on dimension tft_indices_id
    global TFT_Tensor_index;
    global tft_indices;
    adj_mat = sparse(TFT_Tensor_index, length(tft_indices));

    assert(length(varargin)>0, 'generate_tensor_indices_adj_mat_output_first:generate_tensor_indices_adj_mat_output_first', 'generate_tensor_indices_adj_mat_output_first requires at least 1 input tensor')

    % output tensor in first row
    for tftii = 1:length(tft_indices)
        if sum( varargin{1}.index_ids == tft_indices(tftii).id ) == 1
            adj_mat( 1, tft_indices(tftii).id ) = 1;
        end
    end

    for tftii = 1:length(tft_indices)
        for vind = 2:length(varargin)
            if sum( varargin{vind}.index_ids == tft_indices(tftii).id ) == 1
                adj_mat( varargin{vind}.id, tft_indices(tftii).id ) = 1;
            end
        end
    end
end