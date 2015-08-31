function index_inds = get_index_inds_from_ids(index_ids)
    global tft_indices

    tft_indices_ids = [];
    for tftii = 1:length(tft_indices)
        tft_indices_ids = [ tft_indices_ids tft_indices(tftii).id ];
    end

    index_inds = zeros( length( index_ids ) );
    for iiind = 1:length(index_ids)
        index_ids(iiind) = find( tft_indices_ids == index_ids(iiind) );
    end

end