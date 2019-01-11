rand('seed',0);
test_run_data_filename = 'test_random_tucker3_gtp_run_times.mat';

run_times.cardinalities = [];
run_times.default = [];
run_times.steiner = [];
save( 'test_random_tucker3_gtp_run_times.mat', 'run_times' );

for i=1:10
    clear all;
    tft_clear();

    max_cardinality = 30;

    i_ind_card = randi(max_cardinality)+1;
    j_ind_card = randi(max_cardinality)+1;
    k_ind_card = randi(max_cardinality)+1;
    p_ind_card = randi(max_cardinality)+1;
    q_ind_card = randi(max_cardinality)+1;
    r_ind_card = randi(max_cardinality)+1;

    load( 'test_random_tucker3_gtp_run_times.mat', 'run_times' );
    run_times.cardinalities(end+1,:) = [i_ind_card j_ind_card k_ind_card p_ind_card q_ind_card r_ind_card];
    save( 'test_random_tucker3_gtp_run_times.mat', 'run_times' );

    engine_type = 'default';
    run test_random_tucker3_gtp_performance_helper
    load( 'test_random_tucker3_gtp_run_times.mat', 'run_times' );
    eval( ['run_times.' engine_type '(end+1) = ' num2str(run_time) ';']);
    save( 'test_random_tucker3_gtp_run_times.mat', 'run_times' );


    tft_clear();
    engine_type = 'steiner';
    run test_random_tucker3_gtp_performance_helper
    load( 'test_random_tucker3_gtp_run_times.mat', 'run_times' );
    eval( ['run_times.' engine_type '(end+1) = ' num2str(run_time) ';'] );
    save( 'test_random_tucker3_gtp_run_times.mat', 'run_times' );

end

run_times.default ./ run_times.steiner
mean(run_times.default ./ run_times.steiner)