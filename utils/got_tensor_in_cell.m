function [found] = got_tensor_in_cell(tensor, tensor_cell)
    found = sum( cellfun( @(x) x.id == tensor.id, tensor_cell ) );
end