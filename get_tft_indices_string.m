function [tft_indices_str] = get_tft_indices_string(indices)
% given a list of Index objects, returns a comma separated string with their names in base workspace
% 'topic_index, movie_index'
    tft_indices_str = '';
    for i = 1:length(indices)
        if i > 1
            tft_indices_str = [tft_indices_str ', '];
        end
        
        assert( isa( indices{i}, 'Index' ), 'get_tft_indices_string:get_tft_indices_string', ['Arguments must be of type Index, found ' class(indices{i})] );

        tft_indices_str = [tft_indices_str num2str(indices{i}.name) ];
    end
end