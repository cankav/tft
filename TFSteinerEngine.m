classdef TFSteinerEngine < handle
    properties
        kl_divergence;
        base_filename;
        draw_dot;
        config;
        gtp_rule_group_ids; % group ids of each gtp rules, -1 if rule is not a gtp rule
        gtp_group_ids; % unique group ids
        %gtp_inter_group_inds; % indices of gtp operations within their respective groups
    end

    methods
        function obj = TFSteinerEngine(config, gtp_rule_group_ids, base_filename)
            assert( length(gtp_rule_group_ids) == length(config.gtp_rules), 'TFSteinerEngine:TFSteinerEngine', 'Group information must be specified for each gtp rule' );
            non_negative_gtp_rule_group_ids = gtp_rule_group_ids(gtp_rule_group_ids~=-1);
            assert( issorted(non_negative_gtp_rule_group_ids), 'TFSteinerEngine:TFSteinerEngine', 'gtp_rule_group_ids argument must be sorted (excluding -1 values)' );
            obj.gtp_rule_group_ids = gtp_rule_group_ids;
            obj.gtp_group_ids = unique(non_negative_gtp_rule_group_ids);

            % inter_group_index = 1;
            % for i = 1:length(gtp_rule_group_ids)
            %     if gtp_rule_group_ids(i) == -1
            %         inter_group_index = 1;
            %         gtp_inter_group_inds(i) = -1;
            %     else
            %         gtp_inter_group_inds(i) = inter_group_index;
            %         inter_group_index = inter_group_index + 1;
            %     end
            % end

            obj.draw_dot = false;
            if nargin == 3
                if (isunix() == 1 && system('dot -h &> /dev/null') ~= 127) == 0
                    display( 'Steiner tree depends on graphviz (dot) on unix platforms for drawing graphs, graphs will be unavailable' )
                else
                    obj.draw_dot = true;
                    system('if [ ! -d dot_files ]; then mkdir dot_files; fi');
                    obj.base_filename = base_filename;
                end
            end

            obj.config = config;

        end

        function [cost] = get_operation_cost(~, mode, dim_ind, op_state, parent_costs, available_memory)
        % calculate cost of an contract/sum/mult operation over a given dimension in 
        % a given state of the model
        % mem: extra memory usage
        % compute: sum of number of summation and multiplication operation, equals to flintmax if available memory is exceeded
        % lookup: number of data element lookups
        % sum: number of summation operations
        % mult: number of multiplication operations
        % parent_costs: if given added to the calculcated values

            global tft_indices;

            dim_sizes = zeros( 1, length(tft_indices) );
            for di = 1:length(tft_indices)
                dim_sizes(di) = tft_indices(di).cardinality;
            end

            tmp_tensor_dims = full( sum ( op_state( ( op_state(:, dim_ind) ~= 0 ), : ), 1 ) ~= 0 );

            % adj_mat contains observed (output) tensor in first row hence 2:end
            input_tensor_num = size( op_state( op_state(2:end, dim_ind) ~= 0, : ), 1);

            if strcmp(mode, 'contract')
                % memory: size of fill-in tensor
                fill_in_tensor_dims = tmp_tensor_dims;
                fill_in_tensor_dims(1, dim_ind) = 0; % contraction
                t = dim_sizes .* fill_in_tensor_dims;
                t( t==0 ) = 1; % prod identity is 1
                fill_in_tensor_size = prod(t);
                cost.mem = fill_in_tensor_size * 8; % assume double data elements with 8 bytes of storage

                % lookup: for each element of the output we will lookup once 
                % to fetch corresponding element of each input tensor for 
                % cardinality( contracted dimension ) times
                cost.lookup = fill_in_tensor_size * input_tensor_num * dim_sizes(dim_ind);

                % sum: for each element of the output we will perform 
                % cardinality(contracted dimension)-1 number of summations
                cost.sum = fill_in_tensor_size * (dim_sizes(dim_ind)-1);

                % mult: for each element of the output we will perform n-1 number 
                % of multiplications, where n = number of input tensors, for each one of
                % cardinality(contracted dimension) summations
                cost.mult = fill_in_tensor_size * (input_tensor_num-1) * dim_sizes(dim_ind);

            elseif strcmp(mode, 'mult')
                % memory: size of tmp tensor
                %fill_in_tensor_dims = tmp_tensor_dims;
                t = dim_sizes .* tmp_tensor_dims;
                t( t==0 ) = 1; % prod identity is 1
                tmp_tensor_size = prod(t);
                cost.mem = tmp_tensor_size * 8; % assume double data elements with 8 bytes of storage

                % lookup: for each element of the output we will lookup once
                % to fetch corresponding element of each input tensor
                cost.lookup = tmp_tensor_size * input_tensor_num;

                % sum: no summation in this operation mode
                cost.sum = 0;

                % mult: for each element of the output we will perform n-1 number
                % of multiplications, where n = number of input tensors
                cost.mult = tmp_tensor_size * (input_tensor_num-1);

            elseif strcmp(mode, 'sum')
                % memory: size of fill in tensor
                fill_in_tensor_dims = tmp_tensor_dims;
                fill_in_tensor_dims(1, dim_ind) = 0; % contraction
                t = dim_sizes .* fill_in_tensor_dims;
                t( t==0 ) = 1; % prod identity is 1
                fill_in_tensor_size = prod(t);
                cost.mem = fill_in_tensor_size * 8; % assume double data elements with 8 bytes of storage

                % lookup: for each element of the output we will lookup once
                % to fetch corresponding element of the only input tensor for
                % cardinality( contracted dimension) times
                if input_tensor_num ~= 1
                    display([ 'UNEXPECTED NUMBER OF INPUT TENSORS ' num2str(input_tensor_num) ]);
                end
                cost.lookup = fill_in_tensor_size * 1 * dim_sizes(dim_ind);

                % sum: for each element of the output we will perform
                % cardinality( contracted dimension )-1 number of summations
                cost.sum = fill_in_tensor_size * (dim_sizes(dim_ind)-1);

                % mult: no multiplication in this operation mode
                cost.mult = 0;

            end

            % compute: sums up summation and multiplication operations
            cost.compute = cost.sum + cost.mult;


            if nargin > 4
                % add parent costs to current costs
                if parent_costs.mem ~= flintmax % avoid overflow
                    cost.mem = cost.mem + parent_costs.mem;
                    cost.compute = cost.compute + parent_costs.compute;
                    cost.lookup = cost.lookup + parent_costs.lookup;
                    cost.sum = cost.sum + parent_costs.sum;
                    cost.mult = cost.mult + parent_costs.mult;
                end
            end

            if nargin == 6
                % we do not accept solutions requiring more memory
                if cost.mem > available_memory
                    display([ 'get_operation_cost: available memory exceeded with ' num2str(cost.mem) ' on operation over dimension ' tft_indices(dim_ind).name ' available memory ' num2str(available_memory) ]);
                    cost.mem = flintmax;
                    cost.compute = flintmax;
                end
            end
        end

        function [optimal_operations, steiner_tree] = get_optimal_operations(obj, gtp_group_id)
            global tft_indices;

            %global states state_connections costs
            optimal_operations = [];
            states = {};
            state_connections = sparse(0); % from, to
            empty_costs = struct('mem', 0, 'compute', 0, 'lookup', 0, 'sum', 0, 'mult', 0);
            operation_dim_inds = sparse(0); % from, to
            costs = {};
            operation_types = sparse(0); % from, to = c (contraction) / s (sum) / m (mult)
            operation_gtp_inds = sparse(0); % from, to
            state_labels = {};
            edge_labels = containers.Map();
            if isunix
                available_memory = get_total_mem() * 0.8; %get_free_mem()*10; % MB % TODO: x10 virtual memory??
                available_memory = available_memory * 1e6; % bytes
            else
                % TODO: test
                [u,~] = memory;
                available_memory = u.MemAvailableAllArrays; %bytes
            end

            node_offset = 0;
            if obj.draw_dot
                dot_filename = ['dot_files/' obj.base_filename '_' num2str(gtp_group_id) '.dot'];
                svg_filename = ['dot_files/' obj.base_filename '_' num2str(gtp_group_id) '.svg'];
                dot_text = containers.Map();
                dot_text( 'digraph {' ) = 1;
            end

            costs{end+1} = empty_costs;

            gtp_terminals=[];

            gtp_adj_mats = {};
            for gtp_adj_mat_ind = 1:length(obj.config.gtp_rules)
                if obj.gtp_rule_group_ids(gtp_adj_mat_ind) == gtp_group_id
                    output_tensor = obj.config.gtp_rules{gtp_adj_mat_ind}{2};
                    input_tensors = obj.config.gtp_rules{gtp_adj_mat_ind}{3};
                    gtp_adj_mats{end+1} = generate_tensor_indices_adj_mat_output_first( output_tensor, input_tensors{:} );
                end
            end

            for gtp_adj_mat_ind = 1:length(gtp_adj_mats)
                states{1} = gtp_adj_mats{gtp_adj_mat_ind};
                processing_state = 1;
                prev_ig_delta = length(states);
                %display([ 'prev_ig_delta ' num2str(prev_ig_delta) ]);

                if obj.draw_dot
                    if gtp_adj_mat_ind == 1
                        [img_dot, node_offset] = write_dot_svg(states{processing_state}, length(states), node_offset, gtp_group_id, obj.base_filename);
                        dot_text(img_dot) = 1;
                    end
                end

                %display(['while start processing_state ' num2str(processing_state) ' length ' num2str(length(states)) ]);
                while length(states) >= processing_state || processing_state == 1
                    if length(states) > 500
                        display('more than 500 states stoping');
                        break
                    end

                    % if isscalar(states{processing_state})
                    %     display([ char(10) 'processing state ' num2str(processing_state) ' for gtp_adj_mat_ind ' num2str(gtp_adj_mat_ind) ' detour to ' num2str(states{processing_state}) ]);
                    % else
                    %     display([ char(10) 'processing state ' num2str(processing_state) ' for gtp_adj_mat_ind ' num2str(gtp_adj_mat_ind) ]);
                    % end

                    if isscalar( states{processing_state} )
                        % go back to a previously generated state to generate paths for gtp_adj_mat_ind > 1 so that contraction operation can be detected correctly
                        actual_processing_state = processing_state;
                        processing_state = states{actual_processing_state};
                        cur_state = states{processing_state};

                        % prepend observed factor for this loop (states{1} is replaced for each gtp_adj_mat_ind)
                        cur_state = [ states{1}(1,:); cur_state ];

                        % remove factors with same configuration as output factor, these factors should be "frozen" for this detour
                        cur_state_mask = ones(1, size(cur_state,1));
                        for r = 2:size(cur_state,1)
                            if cur_state(r,:) == cur_state(1,:)
                                cur_state_mask(r) = 0;
                            end
                        end
                        cur_state = cur_state( logical(cur_state_mask), : );

                        actual_observed_factor = cur_state(1,:);
                        % display('actual_observed_factor');
                        % display(full(actual_observed_factor));
                        % display('cur_state updated');
                        % display(full(cur_state));
                    else
                        actual_processing_state = 0;
                        cur_state = states{processing_state};
                    end

                    %display( 'starting search' );
                    %display(full(cur_state));

                    observed_dims_mask = ~cur_state(1,:);
                    contraction_dims_mask = repmat( observed_dims_mask, size(cur_state, 1), 1);
                    contraction_dims = find(sum( cur_state .* contraction_dims_mask, 1) ~= 0);

                    for cdi = 1:length(contraction_dims)

                        % stores new states generated by multi input contraction
                        % (1 new state for direct contraction, 1 new state for multiplication-only state)
                        % OR
                        % single input contraction (single new state)

                        new_states = {};
                        new_states_costs = {};
                        new_state_edge_labels = {};
                        new_operation_types = [];
                        % TODO new_states_factor_names = {}

                        % remove contracted factors
                        new_state = cur_state( ~( cur_state(:, contraction_dims(cdi)) ~= 0 ) , : );

                        if( size(cur_state, 1) - size(new_state, 1) == 1 )
                            % update contracted tensor in place
                            new_state = cur_state;

                            % perform contraction
                            new_state( find(new_state(:, contraction_dims(cdi)) ~= 0), contraction_dims(cdi) ) = 0;
                            new_states{end+1} = new_state;
                            new_operation_types(end+1) = 's';

                            new_states_costs{end+1} = obj.get_operation_cost( 'sum', ...
                                                                              contraction_dims(cdi), ...
                                                                              cur_state, ...
                                                                              costs{processing_state}, ...
                                                                              available_memory);

                            new_state_edge_labels{end+1} = [num2str(gtp_adj_mat_ind) 's_' tft_indices(contraction_dims(cdi)).name ' m(' num2str(new_states_costs{end}.mem) ') c(' num2str(new_states_costs{end}.compute)  ')' ];
                        else
                            % new temporary/fill-in tensor needed for multiplication step if more than one tensor will be contracted

                            % start new state using tensors without data on contraction dimension
                            new_state = cur_state( ~( cur_state(:, contraction_dims(cdi)) ~= 0 ) , :);

                            % append new temporary to node as the state without sum step
                            tmp_tensor_dims = full( sum ( cur_state( ( cur_state(:, contraction_dims(cdi)) ~= 0 ), : ), 1 ) ~= 0 );
                            new_state(end+1, :) = tmp_tensor_dims;

                            new_states{end+1} = new_state;
                            new_operation_types(end+1) = 'm';

                            new_states_costs{end+1} = obj.get_operation_cost( 'mult', ...
                                                                              contraction_dims(cdi), ...
                                                                              cur_state, ...
                                                                              costs{processing_state}, ...
                                                                              available_memory);

                            new_state_edge_labels{end+1} = [num2str(gtp_adj_mat_ind) 'm_' tft_indices(contraction_dims(cdi)).name ' m(' num2str(new_states_costs{end}.mem)  ') c(' num2str(new_states_costs{end}.compute) ')' ];



                            % perform sum step and add it as a new state
                            new_state(end, contraction_dims(cdi)) = 0;
                            new_states{end+1} = new_state;
                            new_operation_types(end+1) = 'c';

                            new_states_costs{end+1} = obj.get_operation_cost( 'contract', ...
                                                                              contraction_dims(cdi), ...
                                                                              cur_state, ...
                                                                              costs{processing_state}, ...
                                                                              available_memory);

                            new_state_edge_labels{end+1} = [num2str(gtp_adj_mat_ind) 'c_' tft_indices(contraction_dims(cdi)).name ' m(' num2str(new_states_costs{end}.mem) ') c(' num2str(new_states_costs{end}.compute)  ')' ];

                        end




                        % memoization
                        for nsi = 1:length(new_states)
                            new_state = new_states{nsi};
                            found = 0;
                            for state=1:length(states)
                                % skip detour states, will look at actual states anyways
                                if isscalar( states{state} )
                                    continue
                                end

                                % must have same number of factors
                                if sum( size(new_state) ~= size(states{state}) ) ~= 0 % && ...
                                                                                      %        sum( sum( new_state == states{i} ) ) == numel(new_state)
                                                                                      %    found = i;
                                                                                      %    break;
                                    continue;
                                end

                                f = 0;
                                tmp_state = states{state};
                                for row = 1:size(new_state,1) % skip first factor, the output factor
                                                              % search for row of new_state in states{state}
                                    [~,indx]=ismember(new_state(row,:), tmp_state, 'rows');
                                    if indx == 0
                                        break
                                    end

                                    % problem case given below, must watch indx for twice occurring new state rows
                                    % new_state
                                    %ans =
                                    %     1     0     0     1     0     0
                                    %     1     0     0     1     0     0
                                    %states{state}
                                    %ans =
                                    %     1     0     0     1     0     0
                                    %     1     1     1     1     1     1
                                    % remove matched state from tmp_state so that we avoid the problem given above
                                    tmp_mask = ones(size(tmp_state,1),1);
                                    tmp_mask(indx) = 0;
                                    tmp_state = tmp_state(logical(tmp_mask),:);

                                    f = f + 1;
                                end

                                %display([ 'f ' num2str(f) ' size(new_state,1) ' num2str(size(new_state,1)) ' size(states{i}) ' num2str(size(states{i})) ]);

                                if f == size(new_state,1) && f == size(states{state},1) % +1 for the first factor which may be different due to different target graphs
                                    found = state;
                                    break
                                end

                            end

                            if found
                                % update cost if smaller, different contraction path different cost

                                if new_states_costs{nsi}.mem < available_memory && costs{found}.compute > new_states_costs{nsi}.compute
                                    %display(['found smaller: processing_state ' num2str(processing_state) ' prev value ' num2str(costs{found}.mem) ' new value ' mem_cost ]);
                                    costs{found} = new_states_costs{nsi};
                                end

                                % connect current state to found state
                                state_connections(processing_state, found) = 1;
                                operation_types(processing_state, found) = new_operation_types(nsi);
                                
                                %display(['add found connection ' num2str(processing_state) ' -> ' num2str(found) ]);
                                %display(['processing_state ' gtp_adj_mats(gtp_adj_mat_ind).dims(contraction_dims(cdi)).name]);
                                %display(full( states{processing_state} ));

                                if obj.draw_dot
                                    dot_text( [ 'n' num2str(processing_state) '->' 'n' num2str(found)  '[label="' new_state_edge_labels{nsi} '" color=black ];'] ) = 1;
                                end

                                operation_dim_inds(processing_state, found) = contraction_dims(cdi);
                                operation_gtp_inds(processing_state, found) = gtp_adj_mat_ind;
                                edge_labels([num2str(processing_state) ',' num2str(found)]) = new_state_edge_labels{nsi};

                                % if this is not the first gtp and found state was discovered by a previous gtp_adj_mat_ind
                                % make a visit to previous state for this gtp_adj_mat_ind
                                if gtp_adj_mat_ind ~= 1 && found <= prev_ig_delta
                                    %make sure this detour was not added before
                                    detour_seen_before = false;
                                    for s = prev_ig_delta+1:length(states)
                                        if isscalar( states{s} ) && states{s} == found
                                            detour_seen_before = true;
                                            break;
                                        end
                                    end
                                    if ~detour_seen_before
                                        states{end+1} = found;
                                        costs{end+1} = costs{found};
                                        %display(['insert new detour state ' num2str(length(states)) ]);
                                    else
                                        %display(['NOT re-insert detour state ' num2str(found) ]);
                                        %display( full( states{found} ) );
                                    end
                                end
                            else
                                % append new node
                                %'append'

                                states{end+1} = new_state;
                                costs{end+1} = new_states_costs{nsi};
                                edge_labels([num2str(processing_state) ',' num2str(length(states))]) = new_state_edge_labels{nsi};

                                %display(['set: for state ' num2str(length(states)) ' value ' num2str(mem_cost) ]);
                                state_connections(processing_state, length(states)) = 1;
                                operation_types(processing_state, length(states)) = new_operation_types(nsi);

                                %display(['add new connection ' num2str(processing_state) ' -> ' num2str(length(states)) ]);

                                %display(['processing_state ' gtp_adj_mats(gtp_adj_mat_ind).dims(contraction_dims(cdi)).name]);
                                %display(full( states{processing_state} ));
                                %display('new_state');
                                %display(full(new_state));

                                if obj.draw_dot
                                    dot_text( ['n' num2str(processing_state) '->' 'n' num2str(length(states))  '[label="' new_state_edge_labels{nsi} '" color=black];'] ) = 1;
                                end
                                operation_dim_inds(processing_state, length(states)) = contraction_dims(cdi);
                                operation_gtp_inds(processing_state, length(states)) = gtp_adj_mat_ind;
                                state_labels{processing_state+1} = [ num2str(processing_state+1) ]; %' ' gtp_adj_mats(gtp_adj_mat_ind).dims( contraction_dims(cdi) ).name ' ' num2str(compute_cost)];

                                if obj.draw_dot
                                    [img_dot node_offset] = write_dot_svg( states{length(states)}, length(states), node_offset, gtp_group_id, obj.base_filename, processing_state );
                                    dot_text(img_dot) = 1;
                                end
                                
                            end
                        end % end nsi = 1:length(new_states)
                            
                    end % end cdi = 1:length(contraction_dims);

                    if processing_state == 1
                        processing_state = prev_ig_delta + 1;
                    else
                        if actual_processing_state == 0
                            processing_state = processing_state + 1;
                        else
                            % this was a detour for a gtp_adj_mat_ind > 1, restore processing_state
                            processing_state = actual_processing_state+1;
                        end
                    end
                end % end length(states) >= processing_state

                 %display(['end start processing_state ' num2str(processing_state) ' length ' num2str(length(states)) ]);

                 if ~isscalar(states{end})
                     gtp_terminals(end+1) = length(states);
                 end

            end % end gtp_adj_mat_ind = 1:length(gtp_adj_mats)


             % calculate steiner tree on this graph

             % start with initial state
             steiner_tree = sparse(0); % from, to
             steiner_tree(1,1) = 1;

             % generate edge_weights
             [nnz_rows nnz_cols] = find( state_connections );
             edge_weights = sparse(length(states), length(states));
             for ind = 1:length(nnz_rows)
                 edge_weights( nnz_rows(ind), nnz_cols(ind) ) = costs{ nnz_cols(ind) }.compute;
             end

             unused_terminals = gtp_terminals;

             % loop while steiner tree does not span all terminals
             while true

                 % all vertices included in current steiner tree
                 [~, nc_inds] = find(steiner_tree);
                 nc_inds = unique(nc_inds);

                 % select a terminal x not in T closest to a vertex in T
                 min_dist = flintmax;
                 min_dist_path = [];
                 % try all vertices in current steiner tree
                 for nc_inds_ind = 1:length(nc_inds)
                     % try all available terminals
                     for ut_ind = 1:length(unused_terminals)
                         %display(['find path from ' num2str(nc_inds(nc_inds_ind)) ' to ' num2str(unused_terminals(ut_ind))]);
                         [dist, path] = graphshortestpath(edge_weights, nc_inds(nc_inds_ind), unused_terminals(ut_ind));
                         if dist < min_dist
                             min_dist = dist;
                             min_dist_path = path;
                         end
                     end
                 end

                 % add x to T
                 path_str='';
                 for mdp_ind = 2:length(min_dist_path)
                     steiner_tree(min_dist_path(mdp_ind-1), min_dist_path(mdp_ind)) = 1;
                     path_str = [ path_str ' ' num2str(min_dist_path(mdp_ind-1)) ];
                 end
                 path_str = [ path_str ' ' num2str(min_dist_path(end)) ];
                 %display([ 'add path: ' path_str char(10)]);

                 % remove terminal x from unused_terminals
                 unused_terminals( unused_terminals == min_dist_path(end) ) = [];

                 % stop if we have used all terminals
                 if isempty(unused_terminals)
                     break
                 end

             end


             % generate optimal_operations
             [from_inds, to_inds] = find(steiner_tree);
             for i_ind = 1:length(from_inds)
                 from_ind = from_inds(i_ind);
                 to_ind = to_inds(i_ind);
                 if ~(from_ind == 1 && to_ind == 1)
                     operation.index = operation_dim_inds( from_ind, to_ind );
                     operation.type = operation_types( from_ind, to_ind );
                     operation.gtp_index = operation_gtp_inds( from_ind, to_ind );
                     optimal_operations = [optimal_operations operation];
                 end
             end


             if obj.draw_dot
                 dot_text( '}' ) = 1;
                 % write keys to file
                 cmd = ['if [ -f ' dot_filename ' ]; then rm ' dot_filename '; fi'];
                 system(cmd);
                 dot_text_keys = keys(dot_text);
                 selected_to_inds = [];
                 for i = 1:length(dot_text_keys)
                     txt = dot_text_keys{i};

                     if strfind(txt, 'image')
                         % n106[image="/tmp/gtpmodel_wds106.svg" label="" shape=rectangle color=black];

                         state_num = strtok(txt, '[');
                         state_num = str2num(strtok(state_num, 'n'));
                         % last state does not have any connections -> terminal state || no incoming edges || first state also has no incoming edges
                         if size(state_connections,1)+1 == state_num || nnz(state_connections(state_num, :)) == 0 || state_num == 1
                             txt = strrep(txt, 'black', 'green penwidth=4');
                         end

                     elseif strfind(txt, '->')
                         % n1->n2[label="1m_j m(1890) c(1890)"];
                         [from_state_num, to_state_num] = strtok(txt, '-');
                         from_state_num = str2num(strtok(from_state_num, 'n'));

                         to_state_num = strtok(to_state_num, '[');
                         [~, to_state_num] = strtok(to_state_num, 'n');
                         to_state_num = str2num(strtok(to_state_num, 'n'));

                         % if there are multiple edges arriving to_state select first one
                         if sum( selected_to_inds( selected_to_inds == to_state_num ) ) == 0
                             %display(['check if from ' num2str(from_state_num) ' to ' num2str(to_state_num) ' exists']);
                             if size(steiner_tree,1) >= from_state_num && steiner_tree( from_state_num, to_state_num )
                                 txt = strrep(txt, 'black', 'red penwidth=4');
                                 % make sure we do not arrive this state again, it is already arrived with another edge
                                 selected_to_inds = [selected_to_inds to_state_num];
                             end
                         end
                     end

                     cmd = ['echo '' ' txt ' '' >> ' dot_filename ];
                     system(cmd);
                 end

                 cmd = ['if [ -f ' svg_filename ' ]; then rm ' svg_filename '; fi; cd dot_files && dot -Tsvg ' obj.base_filename '_' num2str(gtp_group_id) '.dot > ' obj.base_filename '_' num2str(gtp_group_id) '.svg ; #display ' svg_filename ];
                 system(cmd);
             end

         end

         function [] = factorize(obj)
             % calculate optimal path in steiner tree for each group

             get_optimal_operations_tic = tic;
             optimal_gtp_operations = [];
             for gg_ind = 1:length(obj.gtp_group_ids)
                 ops = obj.get_optimal_operations(obj.gtp_group_ids(gg_ind));
                 optimal_gtp_operations = [optimal_gtp_operations ops];
                 %display([char(10) 'optimal operations for gtp group id ' num2str(gg_ind)]);
                 %for i = 1:length(ops)
                 %    display_steiner_tree_operation( ops(i) );
                 %end
             end
             display( ['get_optimal_operations took ' num2str(toc(get_optimal_operations_tic)) ' seconds '] );

             total_tic = tic;
             execution_times = zeros(length(obj.config.gtp_rules),1);
             for it_num = 1:obj.config.iteration_number
                 iteration_tic = tic;
                 %display([ char(10) 'iteration ' num2str(it_num) ]);
                 for rule_ind = 1:length(obj.config.gtp_rules)
                     % if iscell( obj.config.gtp_rules{rule_ind}{3} )
                     %     input = num2str(cellfun( @(x) x.id, obj.config.gtp_rules{rule_ind}{3} ));
                     % else
                     %     input = obj.config.gtp_rules{rule_ind}{3};
                     % end
                     %display_rule( obj.config.gtp_rules{rule_ind}, rule_ind, 'Executing ' );

                     execution_tic = tic;
                     if obj.config.gtp_rules{rule_ind}{1} == 'GTP'
                         assert( sum_all_dims(size(obj.config.gtp_rules{rule_ind}{2}.data)) ~= 0, 'TFDefaultEngine:TFDefaultEngine', 'GTP operation requires output tensor with non-zero data' );

                         gtp_group_index = find(obj.gtp_group_ids==obj.gtp_rule_group_ids(rule_ind));
                         group_operations = optimal_gtp_operations(gtp_group_index);
                         %gtp_inter_group_ind = gtp_inter_group_inds(rule_ind);
                         %gtp_operations = group_operations(gtp_inter_group_ind);
                         input_tensors = obj.config.gtp_rules{rule_ind}{3};
                         output_tensor = obj.config.gtp_rules{rule_ind}{2};

                         for operation_ind = 1:length(group_operations)
                             operation = group_operations(operation_ind);
                             if operation.type == 'c'
                                 gtp_mex(16, output_tensor, input_tensors{:} );
                             elseif ops(i).type == 's' || ops(i).type == 'm'
                                 if ops(i).type == 's'
                                     op = '@sum';
                                 elseif ops(i).type == 'm'
                                     op = '@times';
                                 end

                                 for it_ind = 1:length(input_tensors)
                                     if it_ind == 1
                                         output_tensor.data = bsxfun( @eq, output_tensor.data, input_tensors{it_ind}.data );
                                     else
                                         output_tensor.data = bsxfun( eval(op), output_tensor.data, input_tensors{it_ind}.data );
                                     end
                                 end
                             end
                         end

                     else
                         obj.config.gtp_rules{rule_ind}{2}.data = eval( obj.config.gtp_rules{rule_ind}{3} );
                     end

                     execution_times(rule_ind) = execution_times(rule_ind) + toc(execution_tic);
                 end

                 obj.kl_divergence( :, it_num ) = get_kl_divergence_values(obj.config.tfmodel);

                 %display( ['iteration time ' num2str(toc(iteration_tic)) ' seconds divergences ' num2str( obj.kl_divergence( :, it_num )' ) ] );

             end % end iteration
             display( ['operation time ' num2str(toc(total_tic)) ' average execution_times ' num2str((execution_times./obj.config.iteration_number)')] );
         end
    end
end