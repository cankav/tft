function [dot_text node_offset] = write_dot_svg(adj_mat, cluster_id, node_offset, gtp_groups_id, parent_cluster_id)
    if nargin == 5
        [d node_offset] = get_dot_cluster_from_adj_mat( adj_mat, cluster_id, node_offset );
    elseif nargin == 6
        [d node_offset] = get_dot_cluster_from_adj_mat( adj_mat, cluster_id, node_offset, parent_cluster_id );
    end
    system(['echo '' ' d ' '' > /tmp/gtpmodel_wds_group_' num2str(gtp_groups_id) '_cluster' num2str(cluster_id)  '.dot; dot -Tsvg /tmp/gtpmodel_wds_group_' num2str(gtp_groups_id) '_cluster' num2str(cluster_id) num2str(cluster_id) '.dot > /tmp/gtpmodel_wds' num2str(cluster_id) '.svg' ]);
    
    dot_text = ['n' num2str(cluster_id) '[image="/tmp/gtpmodel_wds_group_' num2str(gtp_groups_id) '_cluster' num2str(cluster_id) num2str(cluster_id) '.svg" label="" shape=rectangle color=black];' ];
end
