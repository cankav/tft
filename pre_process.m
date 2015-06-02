function [] = pre_process( )
    global tft_indices
    
    if length(tft_indices) == 0

        % generate tft_indices
        vars = evalin('base', 'whos');
        for var_ind = 1:length(vars)
            if strcmp( vars(var_ind).class, 'Index' )
                tft_indices = [ evalin( 'base', vars(var_ind).name ) tft_indices];
            end
        end

        display( [ 'pre_process: generated tft_indices of length ' num2str( length(tft_indices) ) ] );

        % reshape tensor data
        for var_ind = 1:length(vars)
            if strcmp( vars(var_ind).class, 'Tensor' )

                % make sure dimension of the data are in the same order as in tft_indices
                permute_array=[];
                for tensor_index_ind = 1:evalin('base', [ 'length(' vars(var_ind).name '.indices)' ])
                    missing_index_num=0;
                    for tft_indices_ind = 1:length(tft_indices)
                        if tft_indices(tft_indices_ind).id == evalin('base', [vars(var_ind).name '.indices{' num2str(tensor_index_ind) '}.id'])

                            num_of_tft_indices_not_in_tensor_indices = 0;
                            for tft_ii = 1:tft_indices_ind
                                found = false;
                                for t_ii = 1:evalin('base', [ 'length(' vars(var_ind).name '.indices)' ])
                                    if tft_indices(tft_ii).id == evalin('base', [vars(var_ind).name '.indices{' num2str(t_ii) '}.id'])
                                        found = true;
                                        break;
                                    end
                                end
                                if found == false
                                    num_of_tft_indices_not_in_tensor_indices = num_of_tft_indices_not_in_tensor_indices + 1;
                                end
                            end

                            permute_array = [ permute_array tft_indices_ind-num_of_tft_indices_not_in_tensor_indices ];
                            found=true;
                            break;
                        end
                    end
                end
                cmd = [vars(var_ind).name '.data = permute (' vars(var_ind).name '.data, [' num2str(permute_array) '])']
                evalin('base', cmd)


                % check data size for consitency
            end
        end
    end
end