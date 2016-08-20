function [] = check_divergence(divergence_values)
    for r = 1:size(divergence_values,1)
        assert( issorted(fliplr(divergence_values(r,:))), 'check_divergence:check_divergence', ['Divergence values are not descending for model number ' num2str(r)]);
    end
end