function [] = display_steiner_tree_operation(operation)
    if operation.type == 'c'
        type = 'contract';
    elseif operation.type == 's'
        type = 'sum';
    elseif operation.type == 'm'
        type = 'multiply';
    else
        error('Unknown operation type');
    end

    global tft_indices;
    index_name = tft_indices(operation.index).name;

    display( [type ' over index ' index_name ' for gtp index ' num2str(operation.gtp_index)] );
end