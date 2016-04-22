function [] = pre_process()
    %tic
    global tft_indices
    global TFT_Tensors

    if length(tft_indices) == 0
        TFT_Tensors = {};

        % generate tft_indices
        vars = evalin('base', 'whos');
        for var_ind = 1:length(vars)
            if strcmp( vars(var_ind).class, 'Index' )
                % set name property
                cmd = [vars(var_ind).name '.name = ''' vars(var_ind).name ''';'];
                evalin('base', cmd);

                tft_indices = [ tft_indices evalin( 'base', vars(var_ind).name ) ];
            end
        end
        % order tft_indices by Index.id fields
        tft_indices = tft_indices( [tft_indices.id] );
            
        %display( [ 'pre_process: generated tft_indices of length ' num2str( length(tft_indices) ) ] );

        % reshape tensor data
        for var_ind = 1:length(vars)
            if strcmp( vars(var_ind).class, 'Tensor' )
                % set name property
                cmd = [vars(var_ind).name '.name = ''' vars(var_ind).name ''';'];
                evalin('base', cmd)

                % populate Tensor.index_ids
                tensor_index_len = evalin('base', [ 'length(' vars(var_ind).name '.indices);' ]);
                index_ids = zeros( 1, tensor_index_len );
                for tensor_index_ind = 1:tensor_index_len
                    cmd = [ vars(var_ind).name '.indices{' num2str(tensor_index_ind)  '}.id;' ];
                    index_ids( tensor_index_ind ) = evalin('base', cmd);
                end
                cmd = [vars(var_ind).name '.index_ids = [' num2str(index_ids) '];'];
                evalin('base', cmd)

                % insert tft_indices to tensor structure
                evalin('base', ['global tft_indices ; ' vars(var_ind).name '.tft_indices = tft_indices; ']);
                
                if evalin('base', ['issparse(' vars(var_ind).name '.data)']) == 1
                    % convert sparse matrices into sparse arrays
                    evalin( 'base', ['pre_process_sparse_obj_size = size(' vars(var_ind).name '.data);'] );
                    if evalin('base', [ 'pre_process_sparse_obj_size(2) ~= 1' ] )
                        dim1_index_order = evalin('base', [ 'find( [' num2str(1:length(tft_indices)) '] == ' vars(var_ind).name '.indices{1}.id );']);
                        dim2_index_order = evalin('base', [ 'find( [' num2str(1:length(tft_indices)) '] == ' vars(var_ind).name '.indices{2}.id );']);
                        if dim1_index_order > dim2_index_order
                            evalin('base', [ vars(var_ind).name '.data =  reshape( transpose(' vars(var_ind).name '.data), [ prod(size(' vars(var_ind).name '.data)), 1 ] );']);
                        else
                            evalin('base', [ vars(var_ind).name '.data =  reshape( ' vars(var_ind).name '.data, [ prod(size(' vars(var_ind).name '.data)), 1 ] );']);
                        end

                        cmd = [vars(var_ind).name '.reshaped = 1;'];

                        % TODO: calculate original_indices_permute_array

                        evalin('base', cmd)
                    end

                else
                    TFT_Tensors{ evalin('base', [vars(var_ind).name '.id'] ) } = evalin('base', vars(var_ind).name );

                    % make sure dimension of the data are in the same order as in tft_indices
                    permute_array=[];
                    for tensor_index_ind = 1:tensor_index_len
                        missing_index_num=0;
                        for tft_indices_ind = 1:length(tft_indices)
                            if tft_indices(tft_indices_ind).id == evalin('base', [vars(var_ind).name '.indices{' num2str(tensor_index_ind) '}.id'])

                                num_of_tft_indices_not_in_tensor_indices = 0;
                                for tft_ii = 1:tft_indices_ind
                                    found = false;
                                    for t_ii = 1:tensor_index_len
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

                    % insert missing dimensions
                    reshape_array = [];
                    reshape = false;
                    for tft_indices_ind = 1:length(tft_indices)
                        found = false;
                        for tensor_index_ind = 1:tensor_index_len
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
                        if evalin( 'base', [ 'length(' vars(var_ind).name '.data)' ] ) ~= 0
                            cmd = [vars(var_ind).name '.data = reshape(' vars(var_ind).name '.data, [' num2str(reshape_array) ']);'];
                            evalin('base', cmd)

                            cmd = [vars(var_ind).name '.reshaped = 1;'];
                            evalin('base', cmd)
                        end
                    end

                    % check data size for consitency
                end
            end
        end
    end
    %display( [ ' pre_process time: ' num2str(toc) ] );
end