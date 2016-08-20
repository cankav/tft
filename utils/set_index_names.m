function [] = set_index_names()
    vars = evalin('base', 'whos');
    for var_ind = 1:length(vars)
        if strcmp( vars(var_ind).class, 'Index' ) && ~strcmp( vars(var_ind).name, 'tft_indices' )
            % set name property
            cmd = [vars(var_ind).name '.name = ''' vars(var_ind).name ''';'];
            evalin('base', cmd);
        end
    end
end