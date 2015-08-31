function res = sum_all_dims(data)
    res = data;
    for i = 1:ndims(data)
        res = sum(res);
    end
end