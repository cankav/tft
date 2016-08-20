function [] = display_rule(rule, rule_index, prefix)
    if isstr(rule{3})
        input_str = [ ' ' rule{3} ];
    else
        input_str = '';
        for input_ind = 1:length(rule{3})
            if input_ind > 1
                input_str = [ input_str ',' ];
            end
            input_str = [ input_str ' ' rule{3}{input_ind}.name ];
        end

    end

    if strcmp(rule{1}, '=')
        disp = [char(9) char(9)];
    else
        disp = char(9);
    end
    if length( rule{2}.name ) < 10
        disp2 = [char(9) char(9)];
    else
        disp2 = char(9);
    end

    str = ['Rule ' num2str(rule_index) char(9) 'Type: ' rule{1} disp 'output ' rule{2}.name disp2 'size ' regexprep(num2str(size(rule{2}.data)), '\s*', 'x') char(9) 'input' input_str];
    if nargin == 3
        str = [ prefix str ];
    end
    display(['' str]);
end