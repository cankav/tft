% for 100k data, first replace \t characters with commas un u.data,
% otherwise csvread does not work correctly.
raw=csvread( 'u.data');
ratings=sparse(0); % user, movie -> rating

row_count = size(raw,1);

for row_ind = 1:row_count
    row = raw(row_ind, :);
    ratings( row(1), row(2) ) = row(3);

    if mod(row_ind, 100000) == 0
        display( [ num2str(row_ind / row_count * 100) ' % completed' ] );
    end
end