function [] = pre_process()
    global tft_indices
    
    if length(tft_indices) == 0

        % generate tft_indices
        vars = evalin('base', 'whos');
        for var_ind = 1:length(vars)
            if strcmp( vars(var_ind).class, 'Index' )
                tft_indices = [ evalin( 'base', vars(var_ind).name ) tft_indices];
            end
        end

        %display( [ 'pre_process: generated tft_indices of length ' num2str( length(tft_indices) ) ] );

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

                if diff(permute_array) ~= 1
                    % if permute_array is not consecutive, permute data array
                    cmd = [vars(var_ind).name '.data = permute(' vars(var_ind).name '.data, [' num2str(permute_array) ']);'];
                    evalin('base', cmd)
                    cmd = [vars(var_ind).name '.reshaped = 1;'];
                    evalin('base', cmd)
                end

                % calculate original_indices_permute_array
                original_indices_permute_array = [];
                last_permute_array_val = 1;
                for indices_ind = 1:length(permute_array)
                    for pai = 1:length(permute_array)
                        if permute_array(pai) == last_permute_array_val
                            original_indices_permute_array = [ original_indices_permute_array pai ];
                            last_permute_array_val = last_permute_array_val + 1;
                            break
                        end
                    end
                end

                % insert original_indices_permute_array to tensor structure
                evalin('base', [vars(var_ind).name '.original_indices_permute_array = [ ' num2str(original_indices_permute_array) '];']);

                % insert tft_indices to tensor structure
                evalin('base', ['global tft_indices ; ' vars(var_ind).name '.tft_indices = tft_indices; ']);

                % insert missing dimensions
                reshape_array = [];
                reshape = false;
                for tft_indices_ind = 1:length(tft_indices)
                    found = false;
                    for tensor_index_ind = 1:evalin('base', [ 'length(' vars(var_ind).name '.indices)' ])
                        if tft_indices(tft_indices_ind).id == evalin('base', [vars(var_ind).name '.indices{' num2str(tensor_index_ind) '}.id'])
                            reshape_array = [ reshape_array tft_indices( tft_indices_ind ).cardinality ];
                            found = true;
                            break;
                        end
                    end

                    if found == false
                        reshape_array = [ reshape_array 1 ];
                        reshape = true;
                    end
                end

                if reshape
                    cmd = [vars(var_ind).name '.data = reshape(' vars(var_ind).name '.data, [' num2str(reshape_array) ']);'];
                    evalin('base', cmd)

                    cmd = [vars(var_ind).name '.reshaped = 1;'];
                    evalin('base', cmd)
                end

                % check data size for consitency
            end
        end
    end
end