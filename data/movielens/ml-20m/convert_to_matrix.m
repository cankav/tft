% test data can not be re-distributed without permission
% here is how to get it:

% download:
% http://files.grouplens.org/datasets/movielens/ml-20m.zip into data movielens/movielens/ml-20m folder
% unzip the zip file, make sure u.data is in movielens/movielens/ml-20m folder
% run convert_to_matrix script to produce ratings.mat (this takes long with 20m dataset)

raw=csvread( 'ratings.csv', 1, 0 );
ratings=sparse(0); % user, movie -> rating

row_count = size(raw,1);

for row_ind = 1:row_count
    row = raw(row_ind, :);
    ratings( row(1), row(2) ) = row(3);

    if mod(row_ind, 100000) == 0
        display( [ num2str(row_ind / row_count * 100) ' % completed' ] );
    end
end