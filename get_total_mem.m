function [mem, unit] =get_total_mem()
%Usage: [mem, unit] =get_free_mem()

    [~,out]=system('vmstat -s -S M | grep "total memory"');
    
    mem=sscanf(out,'%f  free memory');

    unit = 'MB' ; 

end
