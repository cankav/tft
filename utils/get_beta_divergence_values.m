function [beta_divergences] = get_beta_divergence_values(tfmodel)
    beta_divergences = zeros(size(tfmodel.coupling_matrix,1),1);
    for fm_ind = 1:2:length(tfmodel.factorization_model)
        X_data = tfmodel.factorization_model{fm_ind}.data;
        v = round(fm_ind/2);
        X_hat_data = tfmodel.X_hat_tensors(v).data;
        %beta_divergence = X_data.^(2-p)./((1-p)*(2-p)) - X_data.*X_hat_data.^(1-p)/(1-p) + X_hat_data.^(2-p)/(2-p);

        p_value = tfmodel.p_vector(v);
        %TODO: add p_value == 2 and 3 cases for faster calculation of divergence
        assert( p_value == 1, 'get_beta_divergence_values', 'only p_value = 1 is implemented')
        if p_value == 1
            if issparse(X_data)
                kl_divergence =  X_data .* spfun(@log, X_data) - X_data .* spfun(@log, X_hat_data) - X_data + X_hat_data;
            else
                kl_divergence =  X_data .* log( X_data ) - X_data .* log(  X_hat_data ) - X_data + X_hat_data;
            end
            beta_divergence = kl_divergence;
        else
            syms p;
            beta_divergence = limit( X_data.^(2-p)./((1-p)*(2-p)) - X_data.*X_hat_data.^(1-p)/(1-p) + X_hat_data.^(2-p)/(2-p), p, p_value);
        end

        beta_divergences( v ) = sum_all_dims(beta_divergence);
    end
end