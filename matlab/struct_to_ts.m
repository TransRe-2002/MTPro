function struct_to_ts(input_file, output_file)
%STRUCT_TO_TS Convert a plain tsStruct .mat file to a MATLAB TS object file.

    loaded = load(input_file);
    tsStruct = find_ts_struct(loaded);

    name = optional_field(tsStruct, 'name', '');
    data = required_field(tsStruct, 'data');
    startTime = required_field(tsStruct, 'startTime');
    zeroTime = optional_field(tsStruct, 'zeroTime', startTime);
    dt = required_field(tsStruct, 'dt');

    tsObj = TS(name, data, startTime, zeroTime, dt);

    fields = {
        'NCh', 'npts', 'endTime', ...
        'clockRef', 'UserData', 'runID', 'chid', ...
        'latitude', 'longitude', 'elevation', ...
        'missVals', 'segments', 'badRec', ...
        'timeStamp', 'timeStampType', 'metadata', 'units'
    };

    for i = 1:numel(fields)
        field = fields{i};
        if isfield(tsStruct, field) && isprop(tsObj, field)
            tsObj.(field) = tsStruct.(field);
        end
    end

    if isempty(tsObj.NCh) || isempty(tsObj.npts)
        [tsObj.NCh, tsObj.npts] = size(tsObj.data);
    end
    if isempty(tsObj.endTime)
        tsObj.endTime = datevec(datenum(tsObj.startTime) + ...
            (tsObj.npts - 1) * (tsObj.dt / 86400));
    end
    if isempty(tsObj.timeStamp)
        tsObj.setTimeStamp();
    end

    save(output_file, 'tsObj', '-v7');
end

function tsStruct = find_ts_struct(loaded)
    if isfield(loaded, 'tsStruct')
        tsStruct = loaded.tsStruct;
        return;
    end

    names = fieldnames(loaded);
    for i = 1:numel(names)
        value = loaded.(names{i});
        if isstruct(value) && isfield(value, 'data') && isfield(value, 'dt')
            tsStruct = value;
            return;
        end
    end

    error('MTProTSConvert:MissingTSStruct', ...
        'input file does not contain a tsStruct');
end

function value = required_field(s, field)
    if ~isfield(s, field)
        error('MTProTSConvert:MissingField', ...
            'tsStruct is missing required field "%s"', field);
    end
    value = s.(field);
end

function value = optional_field(s, field, default_value)
    if isfield(s, field)
        value = s.(field);
    else
        value = default_value;
    end
end
