function [] = generate_test_model(type)
    if strcmpi(type, 'nmf') % X(ij) = Z1(ik) Z2(kj)
        evalin('base', 'run nmf_model');

    elseif strcmpi( type, 'tucker3') %X(ijk) = A(ip) B(jq) C(kr) G(pqr)
        evalin('base', 'run tucker3_model');

    elseif strcmpi( type, 'tucker3_nmf') % X1(ijk) = A(ip) B(jq) C(kr) G(pqr), X2(ij) = A(ip) Z2(pj)
        evalin('base', 'run tucker3_nmf_model');

    else
        error('Unknown factorization model type')
    end

end