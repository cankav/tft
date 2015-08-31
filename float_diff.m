function res = float_diff( A, B )
% returns 1 if A and B are different floating point values
    res = abs(A-B) > 1e-12*max(A,B);
end 