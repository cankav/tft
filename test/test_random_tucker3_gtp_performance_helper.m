i_index = Index(i_ind_card);
j_index = Index(j_ind_card);
k_index = Index(k_ind_card);
p_index = Index(p_ind_card);
q_index = Index(q_ind_card);
r_index = Index(r_ind_card);

X = Tensor( i_index, j_index, k_index );
A = Tensor( i_index, p_index);
B = Tensor( j_index, q_index);
C = Tensor( k_index, r_index);
G = Tensor( p_index, q_index, r_index);

X.data = rand( i_index.cardinality, j_index.cardinality, k_index.cardinality );
A.data = rand( i_index.cardinality, p_index.cardinality );
B.data = rand( j_index.cardinality, q_index.cardinality );
C.data = rand( k_index.cardinality, r_index.cardinality );
G.data = rand( p_index.cardinality, q_index.cardinality, r_index.cardinality );

pre_process();

p = [1];
phi = [1];

factorization_model = {X, {A, B, C, G}};

model = TFModel(factorization_model, p, phi);

config = TFEngineConfig(model, 10);

if strcmp(engine_type, 'default')
    engine = TFDefaultEngine(config, 'gtp_mex');
elseif strcmp(engine_type, 'steiner')
    engine = TFSteinerEngine(config, ...
    [ ...
        1, -1, 1, -1, -1, 1, -1, -1, ...
        2, -1, 2, -1, -1, 2, -1, -1, ...
        3, -1, 3, -1, -1, 3, -1, -1 ...
        4, -1, 4, -1, -1, 4, -1, -1 ...
    ],'gtp_mex');
else
    error('unknown engine type')
end

gtp_rules = model.update_rules();
rule_ind = 1;
display_rule( gtp_rules{rule_ind}, rule_ind, 'Testing ' );
input_tensors = gtp_rules{rule_ind}{3};

tic
gtp_mex(16, gtp_rules{rule_ind}{2}, input_tensors{:} );
run_time=toc;

%display(['run_time ' engine_type ' ' num2str(run_time)])

check_divergence(engine.beta_divergence);