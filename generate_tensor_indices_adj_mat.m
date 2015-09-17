function [adj_mat] = generate_tensor_indices_adj_mat(varargin)
    % adj_mat( tensor_id, tft_indices_id ) is equal to 1 if tensor with tensor_id has data on dimension tft_indices_id
    global TFT_Tensor_index;
    global tft_indices;
    sparse(TFT_Tensor_index, length(tft_indices));

    % for vind = 1:length(varargin)
    %     display( [ 'generate_tensor_indices_adj_mat: varargin ' num2str(vind) ' ' num2str(varargin{vind}.id) ]) ;
    % end

    for tftii = 1:length(tft_indices)
        for vind = 1:length(varargin)
            if sum( varargin{vind}.index_ids == tft_indices(tftii).id ) == 1
                adj_mat( varargin{vind}.id, tft_indices(tftii).id ) = 1;
            end
        end
    end
end