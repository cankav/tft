function res = sum_all_dims(data)
    res = data;
    if strcmp(class(data), 'sym')
        res = sum(reshape(data, prod(size(data)), 1));
    else
        res = sum(data(:));
        % for i = 1:ndims(data) % TODO replace with sum(data(:))
        %     res = sum(res);
        % end
    end
end