function main(mode, input_file, output_file)
%MTPro TS converter command-line entry point.
%
% Usage:
%   mtpro_ts_convert ts2struct input.mat output.mat
%   mtpro_ts_convert struct2ts input.mat output.mat

    if nargin ~= 3
        fprintf(2, 'Usage:\n');
        fprintf(2, '  mtpro_ts_convert ts2struct input.mat output.mat\n');
        fprintf(2, '  mtpro_ts_convert struct2ts input.mat output.mat\n');
        error('MTProTSConvert:InvalidArguments', ...
            'expected exactly 3 arguments: mode input_file output_file');
    end

    mode = char(mode);
    input_file = char(input_file);
    output_file = char(output_file);

    switch lower(mode)
        case {'ts2struct', 'ts_to_struct'}
            ts_to_struct(input_file, output_file);
        case {'struct2ts', 'struct_to_ts'}
            struct_to_ts(input_file, output_file);
        otherwise
            error('MTProTSConvert:InvalidMode', ...
                'unknown mode "%s"; expected ts2struct or struct2ts', mode);
    end
end
