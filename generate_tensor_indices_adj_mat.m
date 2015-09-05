function [] = generate_tensor_indices_adj_mat(varargin)
    global TFT_Tensor_index;
    global tft_indices;
    sparse(TFT_Tensor_index, length(tft_indices));
    for vind = 1:length(varargin)
        display('OSMAN')
        display(varargin{vind}.id
    end
    for tftii = 1:length(tft_indices)
        for vind = 1:length(varargin)
            if sum( varargin{vind}.index_ids == tft_indices(tftii).id ) == 1
                adj_mat( varargin{vind}.id, tft_indices(tftii).id ) = 1;
            end
        end
        % if sum( output_tensor.index_ids == tft_indices(tftii).id ) == 1
        %     adj_mat( varargin{vind}.id, tft_indices(tftii).id ) = 1;
        % end
    end
end