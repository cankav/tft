evalin('base', 'clear all;');
tft_clear();
rand('seed',0);

i_index = Index(20);
j_index = Index(30);
k_index = Index(40);
p_index = Index(50);
q_index = Index(60);
r_index = Index(7);

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

model = TFModel(factorization_model, p, phi)

gtp_rules = model.update_rules();

for rule_ind = 1:length(gtp_rules)
    display_rule( gtp_rules{rule_ind}, rule_ind, 'rule ' );
end

config = TFEngineConfig(model, 10);
config.timeout = 600;
engine = TFDefaultEngine(config, 'gtp_mex');
tic
engine.factorize();
display( ['TFDefaultEngine.factorize time: ' num2str(toc)] );
plot(engine.beta_divergence');
check_divergence(engine.beta_divergence);







evalin('base', 'clear all;');
tft_clear();
rand('seed',0);

i_index = Index(20);
j_index = Index(30);
k_index = Index(40);
p_index = Index(50);
q_index = Index(6);
r_index = Index(7);

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

model = TFModel(factorization_model, p, phi)

gtp_rules = model.update_rules();

for rule_ind = 1:length(gtp_rules)
    display_rule( gtp_rules{rule_ind}, rule_ind, 'rule ' );
end

config = TFEngineConfig(model, 10);

engine = TFSteinerEngine(config, [ ...
    1, -1, 1, -1, -1, 1, -1, -1, ...
    2, -1, 2, -1, -1, 2, -1, -1, ...
    3, -1, 3, -1, -1, 3, -1, -1 ...
    4, -1, 4, -1, -1, 4, -1, -1 ...
], 'steiner_improvement_tucker3_test');

tic
engine.factorize();
display( ['TFSteinerEngine.factorize time: ' num2str(toc)] );

plot(engine.beta_divergence');
check_divergence(engine.beta_divergence);
