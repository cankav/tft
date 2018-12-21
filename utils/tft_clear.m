global tft_indices
tft_indices = [];

global TFT_Tensors
TFT_Tensors = [];

global TFT_Index_index
TFT_Index_index = 0;

global TFT_Tensor_index
TFT_Tensor_index = 0;

% delete all index objects from base workspace
vars = evalin('base', 'whos');
for var_ind = 1:length(vars)
    if strcmp( vars(var_ind).class, 'Index' )
        evalin('base', ['clear ' vars(var_ind).name ';']);
    end
end
