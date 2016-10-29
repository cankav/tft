function [] = check_divergence(divergence_values)
    assert( sum_all_dims( isnan(divergence_values) ) == 0, 'check_divergence:check_divergence', 'Divergance values can not be NaN' );
    assert( sum_all_dims( isinf(divergence_values) ) == 0, 'check_divergence:check_divergence', 'Divergance values can not be Inf' );

    for r = 1:size(divergence_values,1)
        assert( issorted(fliplr(divergence_values(r,:))), 'check_divergence:check_divergence', ['Divergence values are not descending for model number ' num2str(r)]);
    end
end