classdef TS < handle
    % update: function rotate, Hui Wang 2020 01 22 @cumtb
    properties
        NCh
        npts
        data
        startTime
        zeroTime
        endTime
        clockRef
        UserData
        dt
        name
        runID
        chid
        latitude
        longitude
        elevation
        %orientation
        missVals = NaN
        segments   % array giving start/end points of good data segments
        %  for each channel
        badRec
        timeStamp  % array giving time stamps for each data sample
        % to allow for more flexible plotting
        timeStampType = 'years'
        %   add metadata -- not to be direclty used in the generic TS
        %   class, but as a convenient way to carry around metadata from
        %   (e.g., NIMS binary files).   Idea is to copy this when making
        %   derived TS objects, e.g., by decimation
        metadata
        units
    end
    
    methods
        function obj = TS(name,data,startTime,zeroTime,dt,missVals)
            %   simple class constructor ... if any arguments are given, start
            %   filling in TS object properties with argument values
            %
            %   Usage: obj = TS(name,data,startTime,zeroTime,dt,missVals)
            %
            if nargin > 0
                obj.name = name;
            end
            if nargin > 1
                obj.data = data;
                [obj.NCh,obj.npts] = size(data);
            end
            if nargin > 2
                obj.startTime = startTime;
            end
            if nargin > 3
                obj.zeroTime = zeroTime;
            end
            if nargin > 4
                obj.dt = dt;
                obj.setTimeStamp;
                obj.endTime = datevec(datenum(obj.startTime)+(obj.npts-1)*(obj.dt/(86400)));
            end
            if nargin > 5
                obj.missVals = missVals;
            end
        end
        %******************************************************************
        function setTimeStamp(obj,type)
            %   creates the timeStamp array that allows for plotting and
            %   possibly other useful operations
            if nargin < 2
                type = 'years';
            end
            t1 = datenum(obj.zeroTime);
            t2 = datenum(obj.startTime);
            if strcmp(type,'years')
                t0 = (t2-t1)/365.25+obj.zeroTime(1);
                obj.timeStamp  = t0+(0:obj.npts-1)*(obj.dt/(365.25*86400));
            elseif strcmp(type,'days')
                t0 = (t2-t1);
                obj.timeStamp  = t0+(0:obj.npts-1)*(obj.dt/(86400));
            elseif strcmp(type,'hours')
                t0 = (t2-t1)*24;
                obj.timeStamp  = t0+(0:obj.npts-1)*(obj.dt/(3600));
            elseif strcmp(type,'secs')
                t0 = (t2-t1)*86400;
                obj.timeStamp  = t0+(0:obj.npts-1)*(obj.dt);
            else
                warning('Cannot create time stamp array of this type');
                obj.timeStamp = [];
            end
            obj.timeStampType = type;
        end
        %******************************************************************
        function t = getTime(obj)
            %   creates time channel, with sample 0 corresponding to object
            %   zeroTime
            %
            %   Usage: t = getTime(obj)
            %
            nPtsPerDay = 86400/obj.dt;
            t0 =  fix((datenum(obj.startTime) - datenum(obj.zeroTime))*nPtsPerDay);
            t = t0:t0+obj.npts-1;
        end
        %******************************************************************
        function dataMean = demean(obj)
            %   computes means for each channel and subtracts; returns
            %   mean values in dataMean
            %
            %   Usage: dataMean = demean(obj);
            %
            dataMean = zeros(obj.NCh,1);
            for ich = 1:obj.NCh
                temp = obj.data(ich,:);
                dataMean(ich) = mean(temp(~isnan(temp)));
                obj.data(ich,:) = obj.data(ich,:)-dataMean(ich);
            end
        end
        %******************************************************************
        function struc2TS(obj,X,name,latlon)
            
            %   fill a (possibly empty) TS object with contents of
            %   structure X ... this is because some code had been written
            %   using time series stored as structures, not TS objects
            %   Usage: obj = struc2TS(obj,X,name,latlon);
            
            if nargin < 3
                name  = '';
            end
            if nargin == 4;
                obj.latitude = latlon(1);
                obj.longitude = latlon(2);
            end
            obj.data = X.data;
            [obj.NCh,obj.npts] = size(X.data);
            obj.startTime = X.startTime;
            obj.zeroTime = X.zeroTime;
            obj.dt = X.dt;
            obj.name = name;
        end
        %******************************************************************
        function save(obj,filename)
            %   saves TS object as a mat file, with a standard name for the
            %   object
            %   Usage: save(obj,filename);
            tsObj = obj;
            eval(['save ' filename ' tsObj']);
        end
        %******************************************************************
        function writeAscTS(obj,fileName,scales,compact)
            %
            %  Usage: [writeAscTS(obj,fileName,scales)
            %
            %  writes TS object to ascii file <sta>.asc
            %  with clock header information.  clockRef, and clock_zero
            %  are datevec's providing (1) the time of the first
            %  data in the file, which is assumed to be hourly means
            %  and (2) the reference clock zero for defining set numbers
            %  Last line gives scaling factors for each channel; second to
            %  last line gives missing value codes for each channel
            
            %  NOTE: compact option produces files that are not read correctly
            %      by dnff !!!! DON'T USE   !!!!
            
            if nargin==3
                compact = false;
            end
            if compact
                misval = 444444;
                for ich = 1:obj.NCh
                    ind = ~isnan(obj.data(ich,:));
                    x(ich,ind) = obj.data(ich,ind)-median(obj.data(ich,ind));
                end
            else
                misval = 44444444;
                x = obj.data;
            end
            %   find first good data
            fid = fopen(fileName,'w');
            fprintf(fid,'%d\n',obj.dt);
            fprintf(fid,'%d %d %d %d %d %d\n',obj.startTime);
            fprintf(fid,'%d %d %d %d %d %d\n',obj.zeroTime);
            x = round(diag(scales)*x);
            x(isnan(x)) = misval;
            misVals=  misval*ones(obj.NCh,1);
            if compact
                x(x>10000) = misval;
                switch obj.NCh
                    case 1
                        fprintf(fid,'%6d\n',x);
                        fprintf(fid,'%6d\n',misVals);
                        fprintf(fid,'%6d\n',scales);
                    case 2
                        fprintf(fid,'%6d %6d\n',x);
                        fprintf(fid,'%6d %6d\n',misVals);
                        fprintf(fid,'%6d %6d\n',scales);
                    case 3
                        fprintf(fid,'%6d %6d %6d\n',x);
                        fprintf(fid,'%6d %6d %6d\n',misVals);
                        fprintf(fid,'%6d %6d %6d\n',scales);
                    case 4
                        fprintf(fid,'%6d %6d %6d %6d\n',x);
                        fprintf(fid,'%6d %6d %6d %6d\n',misVals);
                        fprintf(fid,'%6d %6d %6d %6d\n',scales);
                    case 5
                        fprintf(fid,'%6d %6d %6d %6d %6d\n',x);
                        fprintf(fid,'%6d %6d %6d %6d %6d\n',misVals);
                        fprintf(fid,'%6d %6d %6d %6d %6d\n',scales);
                end
            else
                switch obj.NCh
                    case 1
                        fprintf(fid,'%9d\n',x);
                        fprintf(fid,'%9d\n',misVals);
                        fprintf(fid,'%9d\n',scales);
                    case 2
                        fprintf(fid,'%9d %9d\n',x);
                        fprintf(fid,'%9d %9d\n',misVals);
                        fprintf(fid,'%9d %9d\n',scales);
                    case 3
                        fprintf(fid,'%9d %9d %9d\n',x);
                        fprintf(fid,'%9d %9d %9d\n',misVals);
                        fprintf(fid,'%9d %9d %9d\n',scales);
                    case 4
                        fprintf(fid,'%9d %9d %9d %9d\n',x);
                        fprintf(fid,'%9d %9d %9d %9d\n',misVals);
                        fprintf(fid,'%9d %9d %9d %9d\n',scales);
                    case 5
                        fprintf(fid,'%9d %9d %9d %9d %9d\n',x);
                        fprintf(fid,'%9d %9d %9d %9d %9d\n',misVals);
                        fprintf(fid,'%9d %9d %9d %9d %9d\n',scales);
                end
            end
            fclose(fid);
        end
        
        %******************************************************************
        function defaultSP(obj,fileRoot,scales,D)
            %
            %  Usage: [defaultSP(obj,fileRoot,scales)
            %
            %  makes a default SP file, no filter info, just accounting
            %  for rescalling when writing output file; everything
            %  undefined set to unit/zero
            if  nargin<4
                D=0.0;
            end
            fileName = [fileRoot '.sp'];
            fid = fopen(fileName,'w');
            %  site ID
            fprintf(fid,'%s\n',obj.name);
            if ~isempty(obj.latitude)
                lat = obj.latitude;
            else
                lat = 0;
            end
            if ~isempty(obj.longitude)
                lon = obj.longitude;
            else
                lon = 0;
            end
            elev = 0;
            %  lat, lon, elev
            fprintf(fid,'%10.4f %10.4f %7.0f\n',[lat,lon,elev]);
            %  coordinate system orientation (express everything in geographic)
            fprintf(fid,'%6.2f\n',D);
            %  number of channels
            fprintf(fid,'%d\n',obj.NCh);
            %  sampling rate: delta t in seconds (need to input, or put into metadata)
            fprintf(fid,'%10.4f\n',obj.dt);
            %  clock offset, linear drift (fossil code!)
            fprintf(fid,'%d %d \n',[0,0]);
            
            %  Now loop over channel system resp
            for ich = 1:obj.NCh
                fprintf(fid,'%s \n',obj.chid{ich});
                nFilt = 0;
                if lower(obj.chid{ich}(2)) == 'x'
                    az = 0;
                else
                    az = 90;
                end
                if upper(obj.chid{ich}(1))=='H'
                    %   magnetic channel
                    fprintf(fid,'%8.2f %8.2f\n',[az 0.0]);
                    fprintf(fid,'%g',1./scales(ich));
                    fprintf(fid,'%2d',nFilt);
                    fprintf(fid,'%s\n','  conversion nT/count, # of filters');
                elseif upper(obj.chid{ich}(1))=='E'
                    %   electric channel
                    fprintf(fid,'%8.2f %8.2f %d %f\n',[1,...
                        az,0,1]);
                    fprintf(fid,'%g',1./scales(ich));
                    fprintf(fid,'%2d',nFilt);
                    fprintf(fid,'%s\n','  conversion mV/count, # of filters');
                else
                    error('Bad system response channel ID')
                end
            end
            
            %  close file
            
            fclose(fid);
        end
        %******************************************************************
        function setBadRec(obj,path,runID)
            %  Usage : setBadRec(obj,path,runID);
            if nargin == 1
                path = './';
            end
            if nargin < 3
                runID = obj.runID;
            end
            obj.badRec = BR(path,runID);
        end
        %******************************************************************
        function badRec2NaN(obj)
            %   if badRec is set, replace all data in marked segments by
            %   NaNs, and (re)run SetSegments
            %    Usage: badRec2NaN(obj)
            t0 = round((datenum(obj.startTime)-...
                datenum(obj.badRec.refTime))*86400/obj.dt);
            for ich = 1:obj.NCh
                for iSeg = 1:obj.badRec.markedSegs{ich}.NSeg
                    i1 = obj.badRec.markedSegs{ich}.Segs(iSeg,1)-t0;
                    i2 = obj.badRec.markedSegs{ich}.Segs(iSeg,2);
                    i1 = max(1,i1);
                    i2 = min(obj.npts,i2);
                    obj.data(ich,i1:i2) = NaN;
                end
            end
            obj.SetSegments
        end
        %******************************************************************
        function time = findTime(obj,n)
            %  Usae: time = findTime(obj,n);
            nPtsPerDay = 86400/obj.dt;
            time = datenum(obj.startTime) + (n-1)/nPtsPerDay;
            time = datevec(time);
        end
        %******************************************************************
        function ind = findInd(obj,time)
            %  Usage: ind = findInd(obj,time);
            nPtsPerDay = 86400/obj.dt;
            ind = round((datenum(time)-datenum(obj.startTime))*nPtsPerDay)+1;
            ind = max(1,ind);
            ind = min(obj.npts,ind);
        end
        %******************************************************************
        function objOut = lpfilter(obj,Flp)
            %  filters obj.y, resulting in high/low passed signals
            objOut =  copy(obj);
            for ich = 1:obj.NCh
                wt = ones(1,obj.npts);
                temp = obj.data(ich,:);
                wt(isnan(temp)) = 0;
                temp(iisnan(temp)) = 0;
                temp =  filter(Flp.Coeff,1,temp);
                wt = filter(Flp.Coeff,1,wt);
                miss = wt<F.maxMissFrac;
                obj.yl = obj.yl./wt;
                obj.yl(miss) = NaN;
                noff = (F.Nfilt+1)/2;
                objOut.data(ich,:) = temp(noff:end);
            end
        end
        %******************************************************************
        function objOut = filter(obj,F)
            %  filters obj.y, resulting in high/low passed signals
            %    might want to refine treatment of missing ...
            objOut =  copy(obj);
            for ich = 1:obj.NCh
                wt = ones(1,obj.npts);
                temp = obj.data(ich,:);
                miss = isnan(temp);
                temp(miss) = 0;
                temp = filter(F.B,F.A,temp);
                temp = filter(F.B,F.A,temp);
                temp = temp(end:-1:1);
                for k = 1:nFiles
                    wFC = copy(wFCtemplate);
                    X = get_one_TS(obj(k));
                    display(['processing site ' X.name])
                    X = X.extractTimeSeg(DateLimits(1,:),DateLimits(2,:));
                    X.zeroTime = zeroTime;
                    %   remove trend, secular variation for long TS
                    X.removeTrend;
                    NYRS = (datenum(DateLimits(2,:))-datenum(DateLimits(1,:)))/365.25;
                    if NYRS>2
                        nSmooth = 24*365;
                        nOffset = 12*365;
                        X.removeSV(nSmooth,nOffset);
                    end
                    %   and then winFT object
                    wFC.TS2FC(X);
                    %   then add header
                    wFC.Header = createSDMheader(obj(k));
                    sta = obj(k).sta;
                    outFile = [FCdir '/' obj(k).sta '_wFC.mat'];
                    
                    eval(['save ' outFile ' wFC sta']);
                end
                
                
                
                temp(miss) = NaN;
                objOut.data(ich,:) = temp;
            end
        end
        %*****************************************************************
        function GDEplot(obj1,ich,obj2)
            %    Usage: plot(obj1,ich,obj2)
            %  simple default plotting: subtract mean and plot all channels
            %  from TS object, with year labels along the x-axis
            %   ich and obj2 are optional.  If present, only channel ich is
            %   ploted; if obj2 is present channel ich is plotted for both
            %   objects, using different colors
            
            %   this is the old plot -- restored for now because I really
            %   don't like the implementation that Anna used -- readable
            %   code always trumps efficiency -- not!
            
            analytic = ~isreal(obj1.data);
            clrs = [.8,0,0;0,.8,0;0,0,.8; 0, 0,0;.7,.7,.7;1 1 0];
            ch = {'Hx','Hy','Hz'};
            figure('Position',[100,100,1000,600]);
            %  start time in years, for plotting axis (not precise, but
            %  good enough!)
            t1 = datenum(obj1.zeroTime);
            t2 = datenum(obj1.startTime);
            t0 = (t2-t1)/365.25+obj1.zeroTime(1);
            T1  = t0+(0:obj1.npts-1)*(obj1.dt/(365.25*86400));
            years=obj1.dt*obj1.npts/365.25/86400;
            if years<1
            T1  = datetime(obj1.startTime):obj1.dt/86400:datetime(obj1.endTime);
            end
            if nargin ==1
                for ich = 1:obj1.NCh
                    X1 = obj1.data(ich,:);
                    xMean = sum(X1(~isnan(X1)))/sum(~isnan(X1));
                    X1 = X1 - xMean;
                    sc = sum(abs(X1(~isnan(X1))))/sum(~isnan(X1));
                    if analytic
                        plot(T1,real(X1),'color',clrs(3,:))
                        hold on
                        plot(T1,imag(X1),'color',clrs(1,:))
                    else
                        plot(T1,X1+(ich-2)*4*sc,'color',clrs(ich,:))
                    end
                    hold on
                end
                ctitle = obj1.runID;
            else
                X1 = obj1.data(ich,:);
                xMean = sum(X1(~isnan(X1)))/sum(~isnan(X1));
                X1 = X1 - xMean;
                if analytic
                    plot(T1,real(X1),'color',clrs(3,:))
                    hold on
                    plot(T1,imag(X1),'color',clrs(3,:),'linestyle','--')
                else
                    plot(T1,X1,'color',clrs(3,:))
                end
                if obj1.NCh <= 3
                    ctitle = [ obj1.runID  ' : ' ch{ich}];
                else
                    ctitle = [ obj1.runID  ' : channel # ' num2str(ich)];
                end
            end
            if nargin > 2
                t1 = datenum(obj2.zeroTime);
                t2 = datenum(obj2.startTime);
                t0 = (t2-t1)/365.25+obj2.zeroTime(1);
                T1  = t0+(0:obj2.npts-1)*(obj2.dt/(365.25*86400));
                X1 = obj2.data(ich,:);
                xMean = sum(X1(~isnan(X1)))/sum(~isnan(X1));
                X1 = X1 - xMean;
                if analytic
                    plot(T1,real(X1),'color',clrs(1,:))
                    plot(T1,imag(X1),'color',clrs(1,:),'linestyle','--')
                else
                    hold on
                    plot(T1,X1,'color',clrs(1,:))
                end
                if obj1.NCh <= 3
                    ctitle = [ obj1.runID  ' : ' ch{ich} ' : ' obj2.runID];
                else
                    ctitle = [ obj1.runID  ' : channel # ' num2str(ich) ' : ' ...
                        obj2.runID];
                end
            end
            set(gca,'fontweight','demi','fontsize',18)
            title(ctitle)
           xlim([T1(1) T1(end)])
        end    % GDEplot
        %******************************************************************
        function [f] = quickplot(obj1,ich,obj2,obj3)
            %    Usage: [f] = quickplot(obj1,ich,obj2,obj3)
            %  simple default plotting: subtract mean and plot all channels
            %  from TS object, with year labels along the x-axis
            %   ich and obj2 are optional.  If present, only channel ich is
            %   ploted; if obj2 is present channel ich is plotted for both
            %   objects, using different colors
            %  Uses "plot" but subtracts the mean of all objects first
            %  to maintain the original objective
            
            %  This is crazy, making copies of what can be huge objects
            %  just to simplify plotting code.   I'll leave it, but only
            %  for now.  I am reinserting my old version, which is useful
            %  for me
            
            myobj1 = obj1.copy;
            myobj1.demean;
            if nargin == 1
                analytic = ~isreal(myobj1.data);
                sc = ones(1,myobj1.NCh);
                for ich = 1:myobj1.NCh
                    X1 = myobj1.data(ich,:);
                    xMean = sum(X1(~isnan(X1)))/sum(~isnan(X1));
                    X1 = X1 - xMean;
                    sc(ich) = sum(X1(~isnan(abs(X1))))/sum(~isnan(abs(X1))); %nanmean(abs(X1));
                    if ~analytic
                        sc(ich) = (ich-2)*4*sc(ich);
                    end
                end
                myobj1.addscalar(sc);
                f = myobj1.plot;
            elseif nargin == 2
                f = myobj1.plot(ich);
            elseif nargin == 3
                myobj2 = obj2.copy;
                myobj2.demean;
                f = myobj1.plot(ich,myobj2);
            elseif nargin == 4
                myobj2 = obj2.copy;
                myobj2.demean;
                myobj3 = obj3.copy;
                myobj3.demean;
                f = myobj1.plot(ich,myobj2,myobj3);
            end
            
        end
         %******************************************************************
        function [f] = subplotchn(obj1,ich)
            %    Usage: [f] = quickplot(obj1,ich)
            %  simple default plotting: plot each channel in each subplot
            %  from TS object, with year labels along the x-axis
            myobj1=obj1.copy;
            myobj1.demean;
            analytic = ~isreal(myobj1.data);
            clrs = [.8,0,0;0,.4,0;0,0,.8; 0, 0,0;.7,.7,.7];
            ch = myobj1.chid;
            f = figure('Position',[100,100,1000,600]);
            %  start time in years, for plotting axis (not precise, but
            %  good enough!)
            if isempty(myobj1.timeStamp)
                obj1.setTimeStamp;
            end
            T1  = obj1.timeStamp;
            if nargin ==1
                for ich = 1:myobj1.NCh
                    X1 = myobj1.data(ich,:);
                    if analytic
                         subplot(myobj1.NCh,1,ich)
                        plot(T1,real(X1),'color',clrs(3,:))
                        hold on
                        plot(T1,imag(X1),'color',clrs(1,:))
                        legend([obj1.name,':',obj1.chid{ich}])
                    else
                       subplot(obj1.NCh,1,ich)
                       plot(T1,X1,'color',clrs(ich,:))
                         legend([obj1.name,':',obj1.chid{ich}])
                    end
                end

            elseif ich>0
                for k=1:length(ich)
                    X1 = myobj1.data(ich(k),:);
                    if analytic
                        subplot(length(ich),1,k)
                        plot(T1,real(X1),'color',clrs(3,:))
                        hold on
                        plot(T1,imag(X1),'color',clrs(3,:),'linestyle','--')
                       legend([obj1.name,':',obj1.chid{ich(k)}])
                    else
                        subplot(length(ich),1,k)
                        plot(T1,X1,'color',clrs(3,:))
                         legend([obj1.name,':',obj1.chid{ich(k)}])
                    end
                end
            end
            allaxes=findobj(gcf,'type','axes');
            set(allaxes,'fontweight','demi','fontsize',14)
        end    % subplotchn
        function f1=plotMT4G(obj1)
            obj1.demean;
            t=datetime(obj1.startTime):obj1.dt/86400:datetime(obj1.endTime);
            if obj1.NCh==7
                figure(1)
                clf;
                subplot(211)
                Bid=contains(obj1.chid,'B') ;
                obj1.demean;
                plot(t,obj1.data(Bid,:))
                lg=legend(obj1.chid(Bid));
                lg.Location='NorthEast';
                lg.Orientation= 'Horizontal';
                lg.Box= 'off';
                ylabel('B (nT)')
                title(obj1.name)
                subplot(212)
                Eid=contains(obj1.chid,'E') ;
                plot(t,obj1.data(Eid,:));
                ylabel('E (mV/km)')
                %title([Site{isite},' -E [',num2str(lat{1}),' N, ',num2str(lon{1}),'E ]'])
                lg=legend(obj1.chid(Eid));
                lg.Location='NorthEast';
                lg.Orientation= 'Horizontal';
                lg.Box= 'off';
            elseif obj1.NCh==4
                figure(1)
                clf;
                plot(t,obj1.data([end-3,end-1],:));
                hold on
                plot(t,obj1.data([end-2,end],:)+10);
                ylabel('E (mV/km)')
                %title([Site{isite},' -E [',num2str(lat{1}),' N, ',num2str(lon{1}),'E ]'])
                lg=legend('Ex1','Ex2','Ey1','Ey2');
                lg.Location='NorthEast';
                lg.Orientation= 'Horizontal';
                lg.Box= 'off';
                title(obj1.name)
            elseif obj1.NCh==3 % for mag only
                figure(1)
                clf;
                plot(t,obj1.data([1:3],:));
                ylabel('B nT')
                %title([Site{isite},' -E [',num2str(lat{1}),' N, ',num2str(lon{1}),'E ]'])
                lg=legend('Bx','By','Bz');
                lg.Location='NorthEast';
                lg.Orientation= 'Horizontal';
                lg.Box= 'off';
                title(obj1.name)
            end
            f1=gcf;
            f1.PaperUnits='inches';
            f1.PaperSize=[8,6];
            set(findall(gcf,'-property','FontSize'),'FontSize',12);
            set( findall(gcf,'Type','line'),'LineWidth',1);
        end
        %******************************************************************
        function [f] = plot(obj1,ich,obj2,obj3)
            %    Usage: [f] = plot(obj1,ich,obj2,obj3)
            %  simple default plotting: plot all channels
            %  from TS object, with year labels along the x-axis
            %  NO LONGER SUBTRACTS MEAN OR RESCALES. USE quickplot FOR THAT.
            %   ich and obj2 are optional.  If present, only channel ich is
            %   ploted; if obj2 is present channel ich is plotted for both
            %   objects, using different colors
            %  29 Nov 2016 added option to also plot obj3 as well as added
            %  a time stamp property that defines the x-axis for plotting;
            %  auto-generated in the constructor from zeroTime (A. Kelbert)
            %   plotting order for 2-3 objects: blue, red, green
            %  use the ich=0 trick to make an empty plot with same configs
            
            analytic = ~isreal(obj1.data);
            clrs = [.8,0,0;0,.4,0;0,0,.8; 0, 0,0;.7,.7,.7];
            ch = obj1.chid;
            ctitle = '';
            f = figure('Position',[100,100,1000,600]);
            %  start time in years, for plotting axis (not precise, but
            %  good enough!)
            if isempty(obj1.timeStamp)
                obj1.setTimeStamp;
            end
            T1  = obj1.timeStamp;
            if nargin ==1
                for ich = 1:obj1.NCh
                    X1 = obj1.data(ich,:);
                    if analytic
                        plot(T1,real(X1),'color',clrs(3,:))
                        hold on
                        plot(T1,imag(X1),'color',clrs(1,:))
                    else
                        plot(T1,X1,'color',clrs(ich,:))
                    end
                    hold on
                end
                ctitle = obj1.runID;
            elseif ich>0
                X1 = obj1.data(ich,:);
                if analytic
                    plot(T1,real(X1),'color',clrs(3,:))
                    hold on
                    plot(T1,imag(X1),'color',clrs(3,:),'linestyle','--')
                else
                    plot(T1,X1,'color',clrs(3,:))
                end
                if obj1.NCh <= 3
                    ctitle = [ obj1.runID  ' : ' ch{ich}];
                else
                    ctitle = [ obj1.runID  ' : channel # ' num2str(ich)];
                end
            end
            if nargin > 2
                if isempty(obj2.timeStamp)
                    obj2.setTimeStamp;
                end
                T1  = obj2.timeStamp;
                X1 = obj2.data(ich,:);
                if analytic
                    plot(T1,real(X1),'color',clrs(1,:))
                    plot(T1,imag(X1),'color',clrs(1,:),'linestyle','--')
                elseBlundy
                    hold on
                    plot(T1,X1,'color',clrs(1,:))
                end
                if obj1.NCh <= 3
                    ctitle = [ obj1.runID  ' : ' ch{ich} ' : ' obj2.runID ' (red) '];
                else
                    ctitle = [ obj1.runID  ' : channel # ' num2str(ich) ' : ' ...
                        obj2.runID];
                end
            end
            if nargin > 3
                if isempty(obj3.timeStamp)
                    obj3.setTimeStamp;
                end
                T1  = obj3.timeStamp;
                X1 = obj3.data(ich,:);
                if analytic
                    plot(T1,real(X1),'color',clrs(2,:))
                    plot(T1,imag(X1),'color',clrs(2,:),'linestyle','--')
                else
                    hold on
                    plot(T1,X1,'color',clrs(2,:))
                end
                if obj1.NCh <= 3
                    ctitle = [ obj1.runID  ' : ' ch{ich} ' : ' obj2.runID ' (red) : ' obj3.runID ' (green) '];
                else
                    ctitle = [ obj1.runID  ' : channel # ' num2str(ich) ' : ' ...
                        obj2.runID ' : ' obj3.runID];
                end
            end
            set(gca,'fontweight','demi','fontsize',18)
                    if obj1.NCh <= 3
                        legend({'Ch1','Ch2','Ch3'})
                    end
           % title(ctitle)
        end    % plot
        %******************************************************************
        function f1 =plotSpec(obj,ich,hf)
            %   qd plot of spectrum, using matlab pwelch, with all defaults
            %   Usage: obj.plotSpec(ich)
             if nargin < 3
               f1= figure;
             else
                 hold on
             end
            temp = obj.data(ich,:)';
            avg = mean(temp(~isnan(temp)));
            temp = temp-avg;
            temp(isnan(temp))=avg;
            [Pxx,F] = pwelch(temp,[],[],[],1./obj.dt);
            %ha=gca;
          %  loglog(F*1000,Pxx)
            semilogy(F*1000,Pxx)
            set(gca,'Fontweight','demi','fontsize',14)
            xlim([1 20])
          % set(gca.YLabel.String=	\surd)
           % title(['Power Spectrum for channel ' obj.chid{ich} ])
        end
        %******************************************************************
        function Yp = fitPeriodic(obj,T,K)
            %   fit and remove signals of a fixed period, including up to
            %        K harmonics; if K is not given, just fit fundamental
            if nargin < 3
                K = 1;
            end
            
            t = (1:obj.npts)'*obj.dt;
            f = (1:K)./T;
            w = 2*pi*f;
            X = zeros(obj.npts,2*K);
            X(:,1:2:end) = cos(t*w);
            X(:,2:2:end) = sin(t*w);
            [q,r] = qr(X,0);
            obj.data(isnan(obj.data))=0;
             Yp = (obj.data*q)*q';
            obj.data = obj.data-Yp;
        end
        %******************************************************************
        function  deSpike(obj,varargin)
            %  Usage: deSpike(obj,varargin)
            ich = 1:obj.NCh;
            method = 'median';
            nMedian = 10000;
            factor = 25;
            n = length(varargin);
%             if mod(n,2)
%                 error('Optional arguments must occur in pairs')
%             end
            for k = 1:2:n
                option = lower(varargin{k});
                switch option
                    case 'channels'
                        ich = varargin{k+1};
                    case 'method'
                        method = varargin{k+1};
                    case 'nmedian'
                        nMedian = varargin{k+1};
                    case 'factor'
                        factor = varargin{k+1};
                    otherwise
                        error('invalid optional arguments for TS.deSpike')
                end
            end
            
            nMedian = min(nMedian,obj.npts);
            switch method
                case 'median'
                    
                    for i = 1:length(ich)
                        d = obj.data(ich(i),:);
                        dGood = ~isnan(d);
                        d = d(dGood);
                        n = length(d);
                        nSeg = floor(n/nMedian);
                        nUse = nSeg*nMedian;
                        res = reshape(d(1:nUse),nMedian,nSeg);
                        dMed = ones(nMedian,1)*median(res,1);
                        res = res-dMed;
                        scale = ones(nMedian,1)*median(abs(res),1);
                        spikes = reshape(abs(res)> factor*scale,nUse,1);
                        dMed = reshape(dMed,nUse,1);
                        d(spikes) = dMed(spikes);
                        obj.data(ich(i),dGood) = d;
                    end
                case 'fillmiss'
                    nSeg = floor(obj.npts/nMedian);
                    nUse = nSeg*nMedian;
                    for i = 1:length(ich)
                        d = obj.data(ich(i),:);
                        if any(isnan(d))
                            dM = median(d(~isnan(d)));
                            
                            miss = find(isnan(d));
                            d(miss) = dM;
                        else
                            miss = [];
                        end
                        res = reshape(d(1:nUse),nMedian,nSeg);
                        dMed = ones(nMedian,1)*median(res,1);
                        res = res-dMed;
                        scale = ones(nMedian,1)*median(abs(res),1);
                        spikes = reshape(abs(res)> factor*scale,nUse,1);
                        dMed = reshape(dMed,nUse,1);
                        obj.data(ich(i),spikes) = dMed(spikes);
                        if ~isempty(miss)
                            obj.data(ich(i),miss) = NaN;
                        end
                    end
                case 'diff'
                    nSeg = floor((obj.npts-1)/nMedian);
                    nUse = nSeg*nMedian;
                    for i = 1:length(ich)
                        dDiff = ...
                            reshape(diff(obj.data(ich(i),1:nUse+1)),nMedian,nSeg);
                        dDiff(isnan(dDiff)) =   0;
                        %   could subtract median for each segment
                        scale = ones(nMedian,1)*median(abs(dDiff),1);
                        dDiff = reshape(dDiff,nUse,1);
                        scale =  reshape(scale,nUse,1);
                        spikes = zeros(obj.npts,1);
                        up = dDiff > factor*scale;
                        down = dDiff < -factor*scale;
                        ind = find((up(1:end-1)&down(2:end))  | ...
                            (up(2:end)&down(1:end-1)))+1;
                        obj.data(ich(i),ind) = ...
                            (obj.data(ich(i),ind-1) + ...
                            obj.data(ich(i),ind+1))/2.;
                    end
                otherwise
                    error('invalid despiking method for TS.deSpike')
            end
            
        end
        %******************************************************************
        function [objDec] = decimate(obj,n,OPTIONS)
            %    this will be a generic decimation function controlled by
            %    OPTIONS, a structure which defines the low-pass filter type,
            %    and any parameters needed.
            %    Usage: [objDec] = decimate(obj,n,OPTIONS)
           
            if nargin == 2
                OPTIONS = struct('type','boxcar','offset',0,'maxFracMissing',.25);
            end
            samplesPerDay = 86400/obj.dt;
            startSample = round((datenum(obj.startTime)-datenum(obj.zeroTime))* ...
                samplesPerDay);
            objDec = TS;
            objDec.name = obj.name;
            objDec.runID = obj.runID;
            objDec.missVals = obj.missVals;
            objDec.latitude = obj.latitude;
            objDec.longitude = obj.longitude;
            objDec.NCh  = obj.NCh;
            objDec.zeroTime = obj.zeroTime;
            objDec.clockRef = obj.clockRef;
            objDec.chid = obj.chid;
            objDec.UserData = obj.UserData;
            objDec.metadata = obj.metadata;
            if rem(n,1)~=0
                objDec.data=interp1(1:obj.npts,obj.data',1:n:obj.npts)';
                objDec.startTime=obj.startTime;
                %objDec.startTime=datevec(datenum(obj.startTime)+n/86400);
                objDec.dt = obj.dt*n;
                objDec.npts =size(objDec.data,2);
                objDec.endTime=objDec.findTime(objDec.npts);
            else
                switch OPTIONS.type
                    case 'gaussfilt'
                        w = exp(-(2*[-n:n]/n).^2);
                        w = w/sum(w);
                        i1 = n-mod(startSample,n)+OPTIONS.offset;
                        %   first possible start point
                        i1 = 1+mod(i1,n);
                        [nch,npts] = size(obj.data);
                        nout = floor(double(npts-i1+1)/n);
                        i2 =n*nout+i1-1;
                        nextra = npts-i2;
                        nout = nout+1;
                        objDec.data = zeros(nch,nout);
                        for k = 1:nch
                            temp = obj.data(k,i1:i2);
                            dataMean = mean(temp(~isnan(temp)));
                            temp = reshape(obj.data(k,i1:i2),n,nout-1)-dataMean;
                            temp1 = [zeros(n,1) temp];
                            temp =  [temp zeros(n,1)];
                            temp = [temp1 ; temp];
                            temp = [temp ; [ temp(1,2:end) 0 ]];
                            temp(1,n-i1+2:n)  = obj.data(k,1:i1-1)-dataMean;
                            temp(end,n+1:n+nextra) = objDec.data(k,end-nextra+1:end)-dataMean;
                            wt  = w'*ones(1,nout);
                            wt(isnan(temp)) = 0;
                            temp(isnan(temp)) = 0;
                            sumWt = sum(wt,1);
                            objDec.data(k,:) = sum(temp.*wt,1)./sumWt+dataMean;
                            objDec.data(k,sumWt<OPTIONS.maxFracMissing) = NaN;
                        end
                        if all(isnan(objDec.data(:,1)))
                            objDec.data = objDec.data(:,2:end);
                            nout = nout-1;
                        end
                        if all(isnan(objDec.data(:,end)))
                            objDec.data = objDec.data(:,1:end-1);
                            nout = nout-1;
                        end
                        objDec.startTime = datevec(double(...
                            datenum(obj.startTime)+(i1-1.)/samplesPerDay));
                        objDec.dt = obj.dt*n;
                        objDec.npts = nout;
                        objDec.endTime=obj.findTime(obj.npts);
                    case 'boxcar'
                        i1 = n-mod(startSample,n)+OPTIONS.offset;
                        i1 = 1+mod(i1,n);
                        
                        [nch,npts] = size(obj.data);
                        nout = floor(double(npts-i1+1)/n);
                        i2 = n*nout+i1-1;
                        objDec.data = zeros(nch,nout);
                        for k = 1:nch
                            temp = reshape(obj.data(k,i1:i2),n,nout);
                            w  = ones(size(temp));
                            w(isnan(temp)) = 0;
                            temp(isnan(temp)) = 0;
                            nGood = sum(w,1);
                            objDec.data(k,:) = sum(temp,1)./nGood;
                            objDec.data(k,nGood<OPTIONS.maxFracMissing) = NaN;
                        end
                        objDec.startTime = datevec(double(...
                            datenum(obj.startTime)+(i1-1.)/samplesPerDay));
                        objDec.dt = obj.dt*n;
                        objDec.npts = nout;
                        objDec.endTime=obj.findTime(obj.npts);
                    case 'subsample'
                        i1 = n-mod(startSample,n)+OPTIONS.offset;
                        i1 = 1+mod(i1,n);
                        
                        [nch,npts] = size(obj.data);
                        nout = floor(double(npts-i1+1)/n);
                        i2 = n*nout+i1-1;
                        objDec.data = zeros(nch,nout);
                        objDec.data = obj.data(:,i1:n:i2);
                        objDec.startTime = datevec(double(...
                            datenum(obj.startTime)+(i1-1.)/samplesPerDay));
                        objDec.dt = obj.dt*n;
                        objDec.npts = nout;
                        objDec.endTime=obj.findTime(obj.npts);
                end
            end    % decimate
        end
        %******************************************************************
        function SetSegments(obj)
            %   Usage: SetSegments(obj)
            %   set segments attributes (dividing TS into segments with no
            %   missing data)
            obj.segments = cell(obj.NCh,1);
            for k = 1:obj.NCh
                [gapStart,gapEnd] = findGaps(obj,k);
                if isempty(gapStart)
                    obj.segments{k} = [1 obj.npts];
                else
                    segStarts = [1 gapEnd+1];
                    segEnds = [gapStart-1 obj.npts];
                    obj.segments{k} = [segStarts' segEnds'];
                    if gapStart(1) == 1
                        obj.segments{k} = obj.segments{k}(2:end,:);
                    end
                    if gapEnd(end) == obj.npts
                        obj.segments{k} = obj.segments{k}(1:end-1,:);
                    end
                end
            end
        end
        %******************************************************************
        function SubtractSegmentMeans(obj)
            %    Usage: SubtractSegmentMeans(obj);
            %    subtracts mean from each segment
            obj.SetSegments
            for ich = 1:obj.NCh
                [nSeg,~] = size(obj.segments{ich});
                for k = 1:nSeg
                    j1 = obj.segments{ich}(k,1);
                    j2 = obj.segments{ich}(k,2);
                    obj.data(ich,j1:j2) = ...
                        obj.data(ich,j1:j2)-mean(obj.data(ich,j1:j2));
                end
            end
        end
        %******************************************************************
        function [gapStart,gapEnd] = findGaps(obj,ICH)
            %  find gaps in a TS object (or subset of channels)
            %  Usage: [gapStart,gapEnd] = findGaps(obj,ICH);
            if nargin == 1
                ICH = 1:obj.NCh;
            end
            good  = 1-any(isnan(obj.data(ICH,:)),1);
            gapStart = find(diff(good) == -1)+1;
            gapEnd = find(diff(good) ==  1);
            ngaps = length(gapStart);
            if ngaps > length(gapEnd)
                gapEnd = [gapEnd length(good)];
            end
        ends
        %******************************************************************
        function [ngaps,ngood,gapLengths] = stats(obj,ICH)
            %  some statistics of a TS object ... about gaps mostly
            %  Usage: [ngaps,ngood,gapLengths] = stats(obj,ICH)
            if nargin == 1
                ICH = 1:obj.NCh;
            end
            good  = 1-any(isnan(obj.data(ICH,:)),1);
            good = [1 good 1];
            gapStart = find(diff(good) == -1);
            gapEnd = find(diff(good) ==  1);
            ngaps = length(gapStart);
            gapLengths = gapEnd-gapStart;
            ngood = sum(good);
        end
        %******************************************************************
        function [good] = goodSegs(obj,navg,p)
            %   find  contiguous sections of "good" data ... i.e., not
            %   marked as missing
            %   Usage: [good] = goodSegs(obj,navg,p);
            if nargin ==2
                p = 1;
            end
            nMin = p*navg;
            good  = 1-any(isnan(obj.data),1);
            n = floor(obj.npts/navg);
            good = reshape(good(1:n*navg),navg,n);
            good = sum(good,1)>= nMin;
        end
        
        %******************************************************************
        function [objMerge] = merge(obj1,obj2,OPTIONS)
            %   combine two compatible TS objects, preferably using samples
            %   from obj1
            %   Usage: [objMerge] = merge(obj1,obj2,OPTIONS);
            
            if nargin == 2
                OPTIONS = [];
            end
            
            if obj1.name ~= obj2.name
                objMerge = -1;
                return
            end
            if obj1.NCh ~= obj2.NCh
                objMerge = -2;
                return
            end
            
            %   first check compatability
            if obj1.dt ~= obj2.dt
                objMerge = -3;
                return
            end
            t1 = datenum(obj1.startTime);
            e1 = t1*86400/obj1.dt+obj1.npts;
            t2 = datenum(obj2.startTime);
            e2 = t2*86400/obj1.dt+obj2.npts;
            tdiff = (round((t1-t2)*86400))/obj1.dt;
            ediff = round(e2-e1);
            if tdiff-floor(tdiff) ~= 0
                objMerge = -4;
                return
            end
            if tdiff < 0
                %   TS#1 starts first
                startTime = obj1.startTime;
                data = obj1.data;
            else
                %   TS#2 starts first ... pad begining of data array with
                %   missing values
                startTime = obj2.startTime;
                temp = ones(obj1.NCh,tdiff)*NaN;
                data = [ temp obj1.data];
            end
            if ediff > 0
                
                %   TS # 2 ends later ... pad end of data array with
                %   missing values
                temp = ones(obj1.NCh,ediff)*NaN;
                data = [data temp];
            end
            
            %   make merged TS object for output, copy data from TS1 into
            %   this object
            objMerge = TS(obj1.name,data,startTime,obj1.zeroTime,...
                obj1.dt,obj1.missVals);
            objMerge.latitude = obj1.latitude;
            objMerge.longitude = obj1.longitude;
            objMerge.chid = obj1.chid;
            objMerge.UserData = obj1.UserData;
            objMerge.metadata = obj1.metadata;
            
            %   move TS#2 data into data
            data = obj2.missVals(1)*ones(obj1.NCh,objMerge.npts);
            i1 = 1+max(-tdiff,0);
            i2 = i1-1+obj2.npts;
            data(:,i1:i2) = obj2.data;
            
            %  patch TS2 into TS1
            for k = 1:obj1.NCh
                temp = 2*(1-isnan(objMerge.data(k,:)));
                temp = temp + (1-isnan(data(k,:)));
                %   temp = 1 if only obj2 is available, = 2 if only obj1,
                %      = 3 if both
                if ~any(temp == 1)
                    objMerge = [];
                    return
                end
                
                if isempty(OPTIONS)
                    %  simplest: just splice in, without worrying about shifts
                    objMerge.data(k,temp==1) = data(k,temp==1);
                else
                    switch OPTIONS.type
                        case 'predicted'
                            %   this is all assuming that there are no missing
                            %    data in the predicted (obj2) time series
                            notDone = true;
                            while notDone
                                good  = 1-any(isnan(objMerge.data),1);
                                gapStart = find(diff(good) == -1)+1;
                                gapEnd = find(diff(good) ==  1);
                                Ngaps = length(gapStart);
                                if Ngaps > length(gapEnd)
                                    gapEnd = [gapEnd length(good)];
                                end
                                gapLengths = 1+gapEnd-gapStart;
                                NgapsRemaining = Ngaps;
                                for j = 1:Ngaps
                                    if gapLengths(j) <= OPTIONS.maxGap
                                        i1 = min(gapStart(j)-gapLengths(j),1);
                                        i2 = max(gapEnd(j)+gapLengths(j),length(good));
                                        temp1 = objMerge.data(k,i1:gapStart(j)-1);
                                        temp2 = objMerge.data(k,gapEnd(j+1):i2);
                                        n1 = sum(isnan(temp1));
                                        n2 = sum(isnan(temp2));
                                        maxMiss1 = OPTIONS.maxFracMiss*length(temp1);
                                        maxMiss2 = OPTIONS.maxFracMiss*length(temp2);
                                        if n1>= maxMiss1 && n2 >= maxMiss2
                                            temp1 = temp1-data(k,i1:gapStart(j)-1);
                                            temp2 = temp2-data(k,gapEnd(j+1):i2);
                                            temp1(isnan(temp1))=0;
                                            temp2(isnan(temp2))=0;
                                            mu = (sum(temp1)+sum(temp2))/(n1+n2);
                                            objMerge.data(k,gapStart(j):gapEnd(j)) ...
                                                = data(k,gapStart(j):gapEnd(j))+mu;
                                            NgapsRemaining = NgapsRemaining-1;
                                        end
                                    end
                                end
                                %   check to see if done: all gaps filled, or
                                %   didn't fill any gaps last time through
                                notDone = NgapsRemaining > 0 && ...
                                    NgapsRemaining ~=Ngaps;
                            end
                        case 'splice'
                            %   splicing a section into a longer time
                            %   series, allowing for possible offsets at
                            %   splices
                            iStrt = find(temp==3,1,'first');
                            iEnd = find(temp==3,1,'last');
                            objMerge.data(k,temp==1) = data(k,temp==1);
                            i1 = max(1,iStrt-OPTIONS.Npts)-1;
                            i2 = i1-1;
                            x1 = objMerge.data(k,i1:i2);
                            i1 = iStrt;
                            i2  = min(iStrt+OPTIONS.Npts-1,objMerge.npts);
                            x2 = objMerge.data(k,i1:i2);
                            m1 = median(x1);
                            m2 = median(x2);
                            s = median(abs([x1-m1 x2-m2]));
                            if abs(m2-m1)>OPTIONS.Nmed*s
                                %   shift spliced data points
                                objMerge.data(k,iStrt:iEnd) = ...
                                    objMerge.data(k,iStrt:iEnd)+(m1-m2);
                            end
                            i1 = max(1,iEnd-OPTIONS.Npts);
                            i2 = iEnd;
                            x1 = objMerge.data(k,i1:i2);
                            i1 = iEnd+1;
                            i2  = min(iEnd+OPTIONS.Npts-1,objMerge.npts);
                            x2 = objMerge.data(k,i1:i2);
                            m1 = median(x1);
                            m2 = median(x2);
                            s = median(abs([x1-m1 x2-m2]));
                            if abs(m2-m1)>OPTIONS.Nmed*s
                                %   shift spliced data points
                                objMerge.data(k,iEnd+1:end) = ...
                                    objMerge.data(k,iEnd+1:end)+(m1-m2);
                            end
                        otherwise
                            error('Not coded for this merge OPTION')
                    end
                end
            end
        end
        %******************************************************************
        function [objDiff] = diff(obj1,obj2,name)
            %   difference two compatible TS objects, preferentially using samples
            %   from obj1
            %   Usage: [objDiff] = diff(obj1,obj2,name)
            
            if nargin < 3
                name = 'difference';
            end
            
            if obj1.NCh ~= obj2.NCh
                objDiff = -2;
                return
            end
            
            %   first check compatability
            if obj1.dt ~= obj2.dt
                objDiff = -3;
                return
            end
            t1 = datenum(obj1.startTime);
            e1 = t1*86400/obj1.dt+obj1.npts;
            t2 = datenum(obj2.startTime);
            e2 = t2*86400/obj1.dt+obj2.npts;
            tdiff = (round((t1-t2)*86400))/obj1.dt;
            ediff = e2-e1;
            if tdiff-floor(tdiff) ~= 0
                objDiff = -4;
                return
            end
            
            if tdiff < 0
                %   TS#1 starts first
                startTime = obj2.startTime; %#ok<*PROP>
                if ediff > 0
                    %   TS #2 ends later
                    data = obj1.data(:,1-tdiff:end)-obj2.data(:,1:end-ediff);
                else
                    data = obj1.data(:,1-tdiff:end+ediff)-obj2.data;
                end
            else
                %   TS#2 starts first ...
                startTime = obj1.startTime;
                if ediff > 0
                    %   TS #2 ends later
                    data = obj1.data-obj2.data(:,tdiff+1:end-ediff);
                else
                    data = obj1.data(:,1:end+ediff)-obj2.data(:,tdiff+1:end);
                end
            end
            %   make merged TS object for output, copy data from TS1 into
            %   this object
            objDiff = TS(name,data,startTime,obj1.zeroTime,...
                obj1.dt,obj1.missVals);
            objDiff.latitude = obj1.latitude;
            objDiff.longitude = obj1.longitude ;
            objDiff.runID = obj1.runID;
            objDiff.metadata = obj1.metadata;
            for i = 1:length(obj1.chid)
                objDiff.chid{i} = [obj1.chid{i} ' Difference'];
            end
            objDiff.UserData = obj1.UserData;
            objDiff.setTimeStamp(obj1.timeStampType);
        end
        %******************************************************************
        function [objOut] = copy(objIn,name)
            %   create a copy of TS object objIn
            %   Usage:   [objOut] = copy(objIn,name)
            %            second argument is optional; if present rename
            %            object
            if nargin == 1
                name = objIn.name;
            end
            objOut = TS(name,objIn.data,objIn.startTime,objIn.zeroTime,...
                objIn.dt,objIn.missVals);
            objOut.timeStamp = objIn.timeStamp;
            objOut.timeStampType = objIn.timeStampType;
            objOut.clockRef = objIn.clockRef;
            objOut.latitude = objIn.latitude;
            objOut.longitude = objIn.longitude;
            objOut.runID = objIn.runID;
            objOut.chid = objIn.chid;
            objOut.UserData = objIn.UserData;
            objOut.metadata = objIn.metadata;
        end
        %******************************************************************
        function [objOut] = extractPart(objIn,npts,name)
            %   extract part of TS object objIn
            %   Usage:   [objOut] = extractPart(objIn,npts,name)
            %            if length(npts)  = 1, extracts first npts
            %               length(npts) = 2 extracts npts(1):npts(2)
            %            last argument is optional; if present rename
            %            object
            if nargin == 2
                name = objIn.name;
            end
            
            if length(npts) == 1
                objOut = TS(name,objIn.data(:,1:npts),objIn.startTime,...
                    objIn.zeroTime,objIn.dt,objIn.missVals);
            else
                nptsPerDay = 86400/objIn.dt;
                startTime = datevec(datenum(objIn.startTime)+...
                    (npts(1)-1)/nptsPerDay);
                objOut = TS(name,objIn.data(:,npts(1):npts(2)),...
                    startTime,objIn.zeroTime,objIn.dt,objIn.missVals);
            end
            objOut.clockRef = objIn.clockRef;
            objOut.latitude = objIn.latitude;
            objOut.longitude = objIn.longitude;
            objOut.runID = objIn.runID;
            objOut.chid = objIn.chid;
            objOut.UserData = objIn.UserData;
            objOut.metaData = objIn.metaData;
        end
        %******************************************************************
        function [objOut] = extractTimeSeg(objIn,Sdate,Edate)
            %   extract part of TS object objIn
            %   Usage:   [objOut] = extractPart(objIn,Sdate,Edate)
            
            t0 = datenum(objIn.startTime);
            t1 = datenum(Sdate);
            t2 = datenum(Edate);
            i1 = round((t1-t0)*86400/objIn.dt+1);
            if i1 <= 0
                i1 = 1;
                t1 = t0;
            end
            i1 = max(1,i1);
            i2 = round((t2-t0)*86400/objIn.dt+1);
            i2 = min(objIn.npts,i2);
            
            objOut = TS(objIn.name,objIn.data(:,i1:i2),datevec(t1),...
                objIn.zeroTime,objIn.dt,objIn.missVals);
            objOut.clockRef = objIn.clockRef;
            objOut.latitude = objIn.latitude;
            objOut.longitude = objIn.longitude;
            objOut.runID = objIn.runID;
            objOut.chid = objIn.chid;
            objOut.UserData = objIn.UserData;
            objOut.metadata = objIn.metadata;
        end
        %******************************************************************
        function [objSum] = plus(obj1,obj2,name)
            %   sum of two compatible TS objects, preferably using samples
            %   from obj1
            %  Usage: [objSum] = plus(obj1,obj2,name);
            
            if nargin < 3
                name = 'sum';
            end
            
            if obj1.NCh ~= obj2.NCh
                objSum = -2;
                return
            end
            
            %   first check compatability
            if obj1.dt ~= obj2.dt
                objSum = -3;
                return
            end
            t1 = datenum(obj1.startTime);
            e1 = t1*86400/obj1.dt+obj1.npts;
            t2 = datenum(obj2.startTime);
            e2 = t2*86400/obj1.dt+obj2.npts;
            tdiff = (round((t1-t2)*86400))/obj1.dt;
            ediff = e2-e1;
            if tdiff-floor(tdiff) ~= 0
                objSum= -4;
                return
            end
            
            if tdiff < 0
                %   TS#1 starts first
                startTime = obj2.startTime;
                if ediff > 0
                    %   TS #2 ends later
                    data = obj1.data(:,1-tdiff:end)-obj2.data(:,1:end-ediff);
                else
                    data = obj1.data(:,1-tdiff:end+ediff)-obj2.data;
                end
            else
                %   TS#2 starts first ...
                startTime = obj1.startTime;
                if ediff > 0
                    %   TS #2 ends later
                    data = obj1.data-obj2.data(:,tdiff+1:end-ediff);
                else
                    data = obj1.data(:,1:end+ediff)+obj2.data(:,tdiff+1:end);
                end
            end
            %   make merged TS object for output, copy data from TS1 into
            %   this object
            objSum = TS(name,data,startTime,obj1.zeroTime,...
                obj1.dt,obj1.missVals);
            objSum.latitude = obj1.latitude;
            objSum.longitude = obj1.longitude ;
            objSum.runID = obj1.runID;
            objSum.chid = obj1.chid;
            objSum.UserData = obj1.UserData;
            objSum.metadata = obj1.metadata;
        end
        %******************************************************************
        function [Uobj] = VecMult(obj,U,name)
            %   multiply TS object by array of (possibly) complex vectors U
            %   Usae: [Uobj] = VecMult(obj,U,name);
            
            [nr,nc] = size(U);
            [NCh,NPts] = size(obj.data);
            if nc > NCh
                error('Size of TS object and vectors are incompatible')
            end
            if nargin < 3
                name = 'U*obj';
            end
            Uobj = copy(obj,name);
            Uobj.data = U*obj.data(1:nc,:);
            Uobj.NCh= nr;
            Uobj.npts = NPts;
        end
        %******************************************************************
        function rescale(obj,scales)
            %   multiply individual channels by scaling factors given in
            %   vector scales
            %   Usage: rescale(obj,scales);
            
            if length(scales) ~= obj.NCh
                error(['Number of scaling factors does not match number of ' ...
                    'channels in TS object']);
            end
            for ich = 1:obj.NCh
                obj.data(ich,:) = obj.data(ich,:)*scales(ich);
            end
        end
        %******************************************************************
        function addscalar(obj,scales)
            %   adjust individual channels by adding a scalar to each
            %   values given in vector scales
            %   Usage: addscalar(obj,scales);
            
            if length(scales) ~= obj.NCh
                error(['Number of scalars to add does not match number of ' ...
                    'channels in TS object']);
            end
            for ich = 1:obj.NCh
                obj.data(ich,:) = obj.data(ich,:)+scales(ich);
            end
        end
        %******************************************************************
        function objOut = addChannels(obj1,obj2)
            %   merge two TS objects by concatenating channels from obj2 to
            %   those of obj1.   Initially coded only for the case where
            %   the # of points, sampling rate and start time are identical
            %   Could generalize this!
            %   Usage: objOut = addChannels(obj1,obj2);
            
            %   first check compatability
            if obj1.dt ~= obj2.dt
                objOut = -1;
                return
            end
            if obj1.startTime ~= obj2.startTime
                objOut = -2;
                return
            end
            if obj1.npts ~=obj2.npts
                objOut = -3;
                return
            end
            %   objects are consistent enough!
            objOut = copy(obj1);
            objOut.data = [objOut.data; obj2.data];
            objOut.NCh = objOut.NCh+obj2.NCh;
            objOut.chid = cell(objOut.NCh,1);
            for k = 1:obj1.NCh
                objOut.chid{k} = obj1.chid{k};
            end
            for k = 1:obj2.NCh
                kk= k + obj1.NCh;
                objOut.chid{kk} = obj2.chid{k};
            end
        end
        
        %******************************************************************
        function [obj] = real(obj)
            %   convert TS object to real
            %   Usage: [obj] = real(obj);
            
            obj.data = real(obj.data);
        end
        %******************************************************************
        function [obj] = analytic(obj)
            %   convert TS object to complex analytic object
            %    Usage: [obj] = analytic(obj);
            temp = real(obj.data)';
            temp(isnan(temp)) = 0;
            temp2 = hilbert(temp).';
            temp2(temp==0) = NaN;
            obj.data=temp2;
        end
        %******************************************************************
        function [result] = rms(obj)
            %   compute rms of each cannel in TS object
            %   Usage: [result] = rms(obj);
            result = zeros(obj.NCh,1);
            for k = 1:obj.NCh
                temp = obj.data(k,:);
                temp = temp(~isnan(temp));
                result(k) = sqrt(temp*temp'/length(temp));
            end
        end
         %******************************************************************
        function removeTrend(obj)
            for ich = 1:obj.NCh
                Y = obj.data(ich,:).';
                X = (1:obj.npts).';
                ind = ~isnan(Y);
                X = X(ind);
                X = [ones(size(X)) X];
                Y = Y(ind);
                b = X\Y;
                obj.data(ich,ind) = obj.data(ich,ind) - (X*b).';
            end
        end
        %******************************************************************
        function [SV,K] = removeSV(obj,nSmooth,nOffset)
            %    fit and subtract smooth low-frequency variations
            %       nSmooth is width of smoothing window, nOffset is offset
            %       between smmothed estimates; these are interpolated to
            %       find values to subtract at all points
            %    Usage: [SV,K] = removeSV(obj,nSmooth,nOffset);
            
            K = 1:nOffset:obj.npts;
            if K(end) < obj.npts
                K = [K obj.npts];
            end
            SV = zeros(obj.NCh,length(K));
            j = 0;
            for k = K
                j = j+1;
                k1 = max(1,k-round(nSmooth/2));
                k2  = min(obj.npts,k+round(nSmooth/2));
                for ich = 1:obj.NCh
                    temp = obj.data(ich,k1:k2);
                    temp = temp(~isnan(temp));
                    if length(temp)>nSmooth/4 || j == 1 || j == length(K)
                        SV(ich,j) = mean(temp);
                    else
                        SV(ich,j) = NaN;
                    end
                end
            end
            %    apply convolution smoother to estimated SV
            B = [ .0705 .1307 .1911 .2154  .1911   .1307   .0705];
            for ich = 1:obj.NCh
                w = 1-isnan(SV(ich,:));
                temp1 = SV(ich,:);
                temp1(isnan(temp1)) = 0;
                temp1 = conv(temp1,B,'same');
                temp2 =  conv(w,B,'same');
                temp1 = temp1./temp2;
                temp1(isnan(SV(ich,:))) = NaN;
                SV(ich,:) = temp1;
            end
            %   subtract interpolated SV from data
            for ich = 1:obj.NCh
                Y = SV(ich,:);
                X = K(~isnan(Y));
                Y = Y(~isnan(Y));
                temp = interp1(X,Y,1:obj.npts,'spline');
                obj.data(ich,:) = obj.data(ich,:)-temp;
            end
        end
        %******************************************************************
        function removePeriodic(obj,f0,bw,nharmonic)
            %   remove a periodic signal, including harmonics
            %
            %   Usage:   removePeriodic(obj,f0,bw,nharmonic)
            %
            %   f0 is fundamental frequency, bw bandwidth (in hz),
            %   nharmonic is max number of harmonics to include
            
            f1 = 1./(obj.dt*obj.npts);
            N = min(floor((f1*obj.npts/2)/f0),nharmonic);
            dBar = nanmean(obj.data,2);
            for ich = 1:obj.NCh
                obj.data(ich,:) = obj.data(ich,:)-dBar(ich);
            end
             for ich = 1:obj.NCh
                 [F,TF] = fillmissing(obj.data(ich,:),'linear','SamplePoints',1:obj.npts);
                 obj.data(ich,:)=F;
             end
            X = fft(obj.data,[],2);
            for n = 1:N
                i1 = floor((n*f0-bw/2)/f1)+1;
                i2 = ceil((n*f0+bw/2)/f1)+1;
                j2 = obj.npts-i1+2;
                j1 = obj.npts-i2+2;
                X(:,i1:i2) = 0;
                X(:,j1:j2) = 0;
            end
            obj.data = ifft(X,[],2);
            for ich = 1:obj.NCh
                obj.data(ich,:) = real(obj.data(ich,:)+dBar(ich));
            end
        end
        %******************************************************************
        function deStep(obj,varargin)
            %   trying to take out steps ... no evidence that this works!
            %   seems to work OK for magnetic fields, but not for electric
            %   ...   with bluer spectrum, more erroneous steps are
            %   identified, and substantial drift at long periods is
            %   introduced
            %   Usage:  deStep(obj,ICH,Nmad);
            %    defaults are now set for observatory processing ...
            type = 'OBS';
            CH = 1:obj.NCh;
            minOffset = 100;
            avgWindow = 1000;
            maxSteps = 20;
            n = length(varargin);
            if mod(n,2)
                error('Optional arguments must occur in pairs')
            end
            for k = 1:2:n
                option = lower(varargin{k});
                switch option
                    case 'channels'
                        CH = varargin{k+1};
                    case 'type'
                        method = varargin{k+1};
                    case 'minoffset'
                        minOffset = varargin{k+1};
                    case 'avgwindow'
                        avgWindow = varargin{k+1};
                    case 'maxsteps'
                        maxSteps = varargin{k+1};
                    otherwise
                        error('invalid optional arguments for TS.deSpike')
                end
            end
            
            for k = 1:length(CH)
                %   string together all segments,
                temp = obj.data(CH(k),:);
                temp2 = temp(~isnan(temp));
                d = abs(diff(temp2));
                %  find largest steps that exceed minimum
                nSteps = 0;
                INDS = zeros(maxSteps,1);
                for iStep = 1:maxSteps
                    [dMax,ind] = max(d);
                    if dMax > minOffset
                        nSteps = nSteps+1;
                        d(ind) = 0;
                        INDS(nSteps) = ind;
                    else
                        INDS = INDS(1:nSteps);
                        break
                    end
                end
                INDS = sort(INDS);
                %    find average value before and after possible steps
                nd = length(d);
                shift = zeros(nd+1,1);
                nSteps = length(INDS);
                INDS1 = max(INDS-avgWindow,1);
                INDS2 = min(INDS+avgWindow+1,nd);
                INDS1(2:end) = max(INDS1(2:end),INDS(1:end-1)+1);
                INDS2(1:end-1) = min(INDS2(1:end-1),INDS(2:end));
                for j = 1:nSteps
                    step = mean(temp2(INDS(j)+1:INDS2(j)))-mean(temp2(INDS1(j):INDS(j)));
                    if abs(step) >=minOffset
                        shift(INDS(j)+1:end) = shift(INDS(j)+1:end)-step;
                    end
                end
                obj.data(CH(k),~isnan(temp)) = obj.data(CH(k),~isnan(temp))+shift.';
            end
        end
        %******************************************************************
        function oneStep(obj,ICH,Nsteps)
            %   another try ... remove biggest n steps from a single
            %   channel
            %   Usage:  oneStep(obj,ICH,Nsteps);
            if nargin == 2
                Nsteps = 1;
            end
            temp = obj.data(ICH,:);
            d = temp(~isnan(temp));
            d1 = d(1);
            d = diff(d);
            for iStep = 1:Nsteps
                [~,ind] = max(abs(d));
                %i1 = max(ind-Nmean,1);
                % i2 = min(ind+Nmean+1,length(dd);
                % dBefore = mean(d(i1:ind);
                % dAfter = mean(d(ind+1:i2);
                d(ind) = 0;
            end
            d = cumsum([d1 d]);
            obj.data(ICH,~isnan(temp)) = d;
        end
      %******************************************************************
        function  obj = rotate(obj,dec)
               % anti-clockwise rotation
            %   rotate geographic coordination to  geomagnetic coordination
            %  also can use to rotate to other coordination system
            if nargin<2
             [~,~,dec]=igrfmagm(obj.elevation,obj.latitude,obj.longitude,obj.startTime(1));
           % dec=-dec;
           dec=deg2rad(dec);
            end
            %matrix of anti-clockwise rotation
            M=[cos(dec) -sin(dec);sin(dec) cos(dec)];
            obj.data(1:2,:) = M*obj.data(1:2,:);
        end
   %******************************************************************
        function  obj = interpNaN(obj)
            % replace NaN with interp1
            %  also can use to rotate to other coordination system
            for k=1:obj.NCh
                 [~,Ind]=find(~isnan(obj.data(k,:)));
                obj.data(k,:)=interp1(Ind,obj.data(k,Ind),1:obj.npts);
            end
        end
   %******************************************************************
    end
end
end