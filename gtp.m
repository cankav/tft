function [] = gtp(output_tensor, varargin)
    assert( isa(output_tensor, 'Tensor'), 'gtp', 'output_tensor must be a Tensor instance' )
    for i = 1:length(varargin)
        assert( isa(varargin{i}, 'Tensor'), 'gtp', 'varargin elements should be Tensor insantences' )
    end

    assert( length(varargin) > 1, 'gtp', 'number of input tensors must be greater than one' )

    tic;
    pre_process();
    display( [ ' pre_process time: ' num2str(toc) ] );

    global tft_indices;

    input_tensor_index_ids = [];
    % initialized with gtp input tensors, updated as contraction operations are performed.
    % contracted tensors are removed, created temporary tensors are added
    current_input_tensors = [];
    for vind = 1:length(varargin)
        input_tensor_index_ids = [input_tensor_index_ids varargin{vind}.index_ids];
        current_input_tensors = [ current_input_tensors varargin{vind}];
    end
    input_tensor_index_ids = unique( input_tensor_index_ids );
    contraction_index_ids = setdiff( input_tensor_index_ids, output_tensor.index_ids );
    contraction_index_inds = get_index_inds_from_ids( contraction_index_ids );

    for cii_ind = 1:length(contraction_index_ids)
        ci_contraction_tensors = [];  % tensors with data on current contraction dimension
        ci_contraction_tensors_cit_indices = []; % cit: current input tensors
        for cit_ind = 1:length(current_input_tensors)
            current_input_tensor = current_input_tensors(cit_ind);
            if sum( contraction_index_ids(cii_ind) == current_input_tensor.index_ids ) > 1
                ci_contraction_tensors = [ ci_contraction_tensors current_input_tensors(cit_ind)];
                ci_contraction_tensors_cit_indices = [ ci_contraction_tensors_cit_indices cit_ind ];
            end
        end

        for ct_ind = 2:length(ci_contraction_tensors)
            % \prod operation
            if length(current_input_tensors) > 2
                % bsxfun # > 2 input contraction tensors into temporary tensor
                tmp_tensor = create_tensor( [ ci_contraction_tensors(1).index_ids ci_contraction_tensors(2).index_ids ] );
                target_tensor = tmp_tensor;

            elseif length(current_input_tensors) == 2
                % bsxfun # = 2 input contraction tensors into output tensor
                target_tensor = output_tensor;

            %else
                % no \prod operation
            end

            if length(ci_contraction_tensors) >= 2
                target_tensor.data = bsxfun( @times, ci_contraction_tensors(1).data, ci_contraction_tensors(2).data );

                % remove 1st and 2nd ci_contraction_tensors from current_input_tensors
                current_input_tensors( ci_contraction_tensors_cit_indices(1) ) = [];
                current_input_tensors( ci_contraction_tensors_cit_indices(2) ) = [];

                % do not add output_tensor to current_input_tensors
                if length(ci_contraction_tensors) > 2
                    current_input_tensors = [current_input_tensors target_tensor];
                end
            end

            % \sum operation
            target_tensor.data = sum( target_tensor.data, contraction_index_inds );
        end
    end
end