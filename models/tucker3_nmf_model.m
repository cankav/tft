i_index = Index(20);
j_index = Index(30);
k_index = Index(40);
p_index = Index(50);
q_index = Index(6);
r_index = Index(7);

X1 = Tensor( i_index, j_index, k_index );
A = Tensor( i_index, p_index);
B = Tensor( j_index, q_index);
C = Tensor( k_index, r_index);
G = Tensor( p_index, q_index, r_index);
X2 = Tensor( i_index, j_index );
Z2 = Tensor( p_index, j_index);

X1.data = rand( i_index.cardinality, j_index.cardinality, k_index.cardinality );
A.data = rand( i_index.cardinality, p_index.cardinality );
B.data = rand( j_index.cardinality, q_index.cardinality );
C.data = rand( k_index.cardinality, r_index.cardinality );
G.data = rand( p_index.cardinality, q_index.cardinality, r_index.cardinality );
X2.data = rand( i_index.cardinality, j_index.cardinality  );
Z2.data = rand( p_index.cardinality, j_index.cardinality  );

pre_process();

p = [1 1];
phi = [1 1];

factorization_model = {X1, {A, B, C, G}, X2, {A, Z2}};

% run GCTF update rules to estimate latent tensors
%model = TFModel(factorization_model, p, phi);
%config = TFEngineConfig(model, 10);
%engine = TFDefaultEngine(config, 'gtp_mex');
%engine.factorize();
%plot(engine.beta_divergence');
%check_divergence(engine.beta_divergence);