function [dot_text node_offset] = get_dot_cluster_from_adj_mat(adj_mat, cluster_id, node_offset, parent_cluster_id)
% adj_mat: rows are factors, cols are dimensions, first row is observed, rest are latent factors ordered in obj.factors, see get_adjacency_matrix
% inter_cluster_text is not used, can be removed

    global tft_indices;

%dot_text = ['subgraph cluster' num2str(cluster_id) ' { ' char(10)];
    dot_text = ['graph {'];

    factor_inds = containers.Map('KeyType','int32','ValueType','int32');
    dim_inds = containers.Map('KeyType','int32','ValueType','int32');
    [nnz_rows nnz_cols] = find( adj_mat );

    for nind = 1:length(nnz_rows)
        factor_ind = nnz_rows(nind);
        dim_ind = nnz_cols(nind);

        % write factor name if did not before
        if ~isKey( factor_inds, factor_ind )
            node_offset = node_offset + 1;
            dot_text = [ dot_text 'n' num2str(node_offset) ';' char(10)];
            %dot_text = [ dot_text 'n' num2str(node_offset) '[label="' factors(factor_ind).name '", shape="rectangle" ];' char(10)];
            dot_text = [ dot_text 'n' num2str(node_offset) '[label="", fillcolor="black", width=0.4, height=0.3, style="filled", shape="rectangle" ];' char(10)];
            

            factor_inds( factor_ind ) = node_offset;
        end

        % write dim name if did not before
        if ~isKey( dim_inds, dim_ind )
            node_offset = node_offset + 1;
            dot_text = [ dot_text 'n' num2str(node_offset) ';' char(10)];
            dot_text = [ dot_text 'n' num2str(node_offset) '[label="' tft_indices(dim_ind).name '", ];' char(10)];
            dim_inds( dim_ind ) = node_offset;
        end

        % make connection
        dot_text = [ dot_text 'n' num2str(factor_inds(factor_ind))  ' -- ' 'n' num2str(dim_inds( dim_ind )) ';' char(10) ];
        %end
    end

    dot_text = [ dot_text '}' ];
end
