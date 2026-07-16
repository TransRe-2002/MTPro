function ts_to_struct(input_file, output_file)
%TS_TO_STRUCT Convert a MATLAB TS object .mat file to a plain tsStruct file.

    loaded = load(input_file);
    tsObj = find_ts_object(loaded);

    tsStruct = struct();
    fields = {
        'NCh', 'npts', 'data', ...
        'startTime', 'zeroTime', 'endTime', ...
        'clockRef', 'UserData', 'dt', 'name', 'runID', 'chid', ...
        'latitude', 'longitude', 'elevation', ...
        'missVals', 'segments', 'badRec', ...
        'timeStamp', 'timeStampType', 'metadata', 'units', ...
        ''
    };

    for i = 1:numel(fields)
        field = fields{i};
        if isprop(tsObj, field)
            tsStruct.(field) = tsObj.(field);
        end
    end

    save(output_file, 'tsStruct', '-v7');
end

function tsObj = find_ts_object(loaded)
    if isfield(loaded, 'tsObj')
        tsObj = loaded.tsObj;
        return;
    end

    names = fieldnames(loaded);
    for i = 1:numel(names)
        value = loaded.(names{i});
        if isa(value, 'TS')
            tsObj = value;
            return;
        end
    end

    error('MTProTSConvert:MissingTSObject', ...
        'input file does not contain a TS object');
end
