# Tensor Factorization Toolbox

test_tft.m explains all current functionality.

In order to run mex trials, you must run either one of the following compile_mex.m or compile_mex_sparse.m.

# Current Features

- Generalized Tensor Product (GTP) operation by using a single full tensor.
- GTP operation by using temporary tensors.
- GTP operation by iterating over necessary input elements for each output element (parallel mex implementation).
- GTP operation by iterating over necessary sparse input elements for each sparse output element (serial mex implementation).
