function []=setup_tft(varargin)

    if nargin == 0
        prefix = '';
    elseif nargin == 1
        prefix = varargin{1};
    end

    addpath(fullfile(prefix,'core'), ...
            fullfile(prefix, 'engines'), ...
            fullfile(prefix, 'models'), ...
            fullfile(prefix, 'test'), ...
            fullfile(prefix, 'utils') )

    evalin('base', 'TFT_EPS=0.000001;');
end