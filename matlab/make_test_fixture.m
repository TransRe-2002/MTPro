function make_test_fixture(src_mat, out_mat, n_keep, variant)
%MAKE_TEST_FIXTURE Generate a small serialized-TS-object .mat fixture.
%
%   make_test_fixture(src_mat, out_mat, n_keep, variant)
%
%   Loads a plain tsStruct file, keeps the first n_keep samples of every
%   channel, converts it to a TS object via struct_to_ts and saves the
%   serialized object to out_mat.  Used to create the pytest fixtures
%   python/tests/fixtures/ts_object*.mat — files scipy.io cannot parse
%   directly, exercising the pure-Python MCOS reader (io_utils/mcos_reader.py)
%   and the optional mtpro_ts_convert fallback.
%
%   variant:
%     'default'    (default) plain conversion
%     'nondefault' additionally sets properties that have class defaults
%                  (missVals/timeStampType/...) so the MCOS classdef string
%                  table takes a different layout — guards the name-based
%                  property mapping in the Python parser.
%
%   Examples:
%       make_test_fixture('../python/tests/test1.mat', ...
%                         '../python/tests/fixtures/ts_object.mat', 2000)
%       make_test_fixture('../python/tests/test1.mat', ...
%                         '../python/tests/fixtures/ts_object_nondefault.mat', ...
%                         2000, 'nondefault')

    if nargin < 3 || isempty(n_keep)
        n_keep = 2000;
    end
    if nargin < 4 || isempty(variant)
        variant = 'default';
    end

    loaded = load(src_mat);
    if ~isfield(loaded, 'tsStruct')
        error('MakeTestFixture:NoTSStruct', ...
            'source file "%s" does not contain a tsStruct variable', src_mat);
    end
    ts = loaded.tsStruct;
    if n_keep >= ts.npts
        error('MakeTestFixture:TooManyPoints', ...
            'n_keep (%d) must be smaller than npts (%d)', n_keep, ts.npts);
    end

    ts.data = ts.data(:, 1:n_keep);
    ts.npts = n_keep;
    % 清空派生字段，让 struct_to_ts 按截取后的数据重新计算
    ts.endTime = [];
    ts.timeStamp = [];

    tmp_in = [tempname '.mat'];
    tsStruct = ts; %#ok<NASGU> % struct_to_ts 按变量名 tsStruct 查找
    save(tmp_in, 'tsStruct');
    cleanup = onCleanup(@() delete_if_exists(tmp_in));

    struct_to_ts(tmp_in, out_mat);

    if strcmpi(variant, 'nondefault')
        % 重新加载对象并给带默认值的属性赋非默认值，
        % 使 MCOS 序列化的字符串表布局与 default 版不同
        obj_loaded = load(out_mat);
        tsObj = obj_loaded.tsObj;
        tsObj.missVals = 9999;
        tsObj.timeStampType = 'days';
        tsObj.elevation = 1234.5;
        tsObj.badRec = [10 20];
        save(out_mat, 'tsObj', '-v7');
    end

    fprintf('Wrote TS-object fixture (%s): %s\n', variant, out_mat);
end

function delete_if_exists(path)
    if exist(path, 'file')
        delete(path);
    end
end
