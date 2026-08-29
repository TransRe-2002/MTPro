function main(varargin)
%MTPro TS converter command-line entry point.
%
% Usage:
%   mtpro_ts_convert input.mat output.mat
%       Auto-detects the input format (serialized TS object or plain
%       tsStruct) and converts it to the other format.
%   mtpro_ts_convert ts2struct input.mat output.mat
%   mtpro_ts_convert struct2ts input.mat output.mat
%
% Build standalone executable:
%   mcc -m main.m -o mtpro_ts_convert -a TS.m -a struct_to_ts.m -a ts_to_struct.m

    args = varargin;
    if numel(args) == 2
        input_file = char(args{1});
        output_file = char(args{2});
        mode = detect_mode(input_file);
        fprintf('Auto-detected input format: converting %s\n', mode);
    elseif numel(args) == 3
        mode = lower(char(args{1}));
        input_file = char(args{2});
        output_file = char(args{3});
        switch mode
            case {'ts2struct', 'ts_to_struct'}
                mode = 'ts2struct';
            case {'struct2ts', 'struct_to_ts'}
                mode = 'struct2ts';
            otherwise
                error('MTProTSConvert:InvalidMode', ...
                    'unknown mode "%s"; expected ts2struct or struct2ts', mode);
        end
    else
        fprintf(2, 'Usage:\n');
        fprintf(2, '  mtpro_ts_convert input.mat output.mat\n');
        fprintf(2, '      (auto-detects TS object vs tsStruct input)\n');
        fprintf(2, '  mtpro_ts_convert ts2struct input.mat output.mat\n');
        fprintf(2, '  mtpro_ts_convert struct2ts input.mat output.mat\n');
        error('MTProTSConvert:InvalidArguments', ...
            'expected 2 arguments (auto mode) or 3 arguments (explicit mode)');
    end

    switch mode
        case 'ts2struct'
            ts_to_struct(input_file, output_file);
        case 'struct2ts'
            struct_to_ts(input_file, output_file);
    end
    fprintf('Done: %s -> %s\n', input_file, output_file);
end

function mode = detect_mode(input_file)
%DETECT_MODE Inspect a .mat file and decide the conversion direction.
%   A file containing a serialized TS object is converted to tsStruct;
%   a file containing a plain tsStruct is converted to a TS object.
    loaded = load(input_file);
    names = fieldnames(loaded);
    has_ts_object = false;
    has_ts_struct = false;
    for i = 1:numel(names)
        value = loaded.(names{i});
        if isa(value, 'TS')
            has_ts_object = true;
        elseif isstruct(value) && isfield(value, 'data') && isfield(value, 'dt')
            has_ts_struct = true;
        end
    end

    if has_ts_object
        mode = 'ts2struct';
    elseif has_ts_struct
        mode = 'struct2ts';
    else
        error('MTProTSConvert:UnknownFormat', ...
            'input file "%s" contains neither a TS object nor a tsStruct', ...
            input_file);
    end
end
