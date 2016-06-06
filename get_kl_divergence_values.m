function [kl_divergences] = get_kl_divergence_values(tfmodel)
    kl_divergences = zeros(size(tfmodel.coupling_matrix,1),1);
    for fm_ind = 1:2:length(tfmodel.factorization_model)
        X_data = tfmodel.factorization_model{fm_ind}.data;
        v = round(fm_ind/2);
        X_hat_data = tfmodel.X_hat_tensors(v).data;
        kl_divergence =  X_data .* log( X_data ) - X_data .* log(  X_hat_data ) - X_data + X_hat_data;
        kl_divergences( v ) = sum_all_dims(kl_divergence);
    end
end