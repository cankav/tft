clear all;
tft_clear();
rand('seed',0);
global tft_indices

movie_index = Index(1312);
user_index = Index(1384);
genre_index = Index(19);
%movie_index=Index(2);
%user_index=Index(3);
%genre_index=Index(4);

% 131262x1x19
% 18178868166x1 -> sparse with 20000263 elements
% 1x138493x19

output_tensor = Tensor(movie_index, genre_index);
output_tensor.data = rand(movie_index.cardinality, genre_index.cardinality);

input_tensor_1 = Tensor(movie_index, user_index);
input_tensor_1.data = sparse(movie_index.cardinality, user_index.cardinality);

input_tensor_2 = Tensor(user_index, genre_index);
input_tensor_2.data = rand(user_index.cardinality, genre_index.cardinality);

pre_process();

gtp_mex_time = tic;
gtp_mex(16, output_tensor, input_tensor_1,input_tensor_2);
display( [ 'trial time: ' num2str(toc(gtp_mex_time)) ] );
