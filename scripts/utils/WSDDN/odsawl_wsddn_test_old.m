function [detsTrainval] = odsawl_wsddn_test(datasetPath,expPath,imdb,gpus,...
	varargin)
%test script for WSDDN in ODSAWL
%	author @G.Y.Guo


opts.expDir = expPath ;
opts.dataDir = datasetPath;
% if you have limited gpu memory (<6gb), you can change the next 2 params
opts.maxNumProposals = inf; % limit number
opts.imageScales = [480,576,688,864,1200]; % scales

opts.train.gpus = gpus ;
opts.train.prefetch = true ;

opts.numFetchThreads = 1 ;
opts = vl_argparse(opts, varargin) ;

display(opts);
if ~exist(fullfile(datasetPath,'VOCdevkit','VOCcode','VOCinit.m'),'file')
  error('VOCdevkit is not installed');
end
addpath(fullfile(datasetPath,'VOCdevkit','VOCcode'));
opts.train.expDir = opts.expDir ;


%detsOnTestPath = fullfile(expPath,'detsTest.mat');
detsOnTrainvalPath = fullfile(expPath,'detsTrainval.mat');
%apsOnTestPath = fullfile(expPath,'apsTest.txt');
apsOnTrainvalPath = fullfile(expPath,'apsTrainval.txt');

if ~exist(detsOnTrainvalPath)

	% Network initialization
	net = load(fullfile(opts.expDir,'net-epoch-20.mat'));
	net = net.net
	net = dagnn.DagNN.loadobj(net) ;

	%Database initialization
	minSize = 20;
	imdb = fixBBoxes(imdb, minSize, opts.maxNumProposals);
	
	%Detect
	% [detsTest] = test(opts, net, imdb, 'test',detsOnTestPath,apsOnTestPath);
	% fprintf('wsddn test on test done\n');
	
	[detsTrainval] = test(opts, net, imdb, 'trainval',detsOnTrainvalPath,apsOnTrainvalPath);
	fprintf('wsddn test on trainval done\n');
	
else
    detsTrainval = load(detsOnTrainvalPath);
end




% --------------------------------------------------------------------
function [dets] = test(opts, net, imdb, testset,detsPath,apsPath)
% --------------------------------------------------------------------
%set test 
if strcmp(testset, 'test')
	testIdx = find(imdb.images.set == 3);
elseif strcmp(testset, 'trainval')
	imdb.images.set(imdb.images.set == 2) = 1;
	testIdx = find(imdb.images.set == 1);
else
	error('no testst %s', testset);
end

if ~exist(detsPath)
	net.mode = 'test' ;
	if ~isempty(opts.train.gpus)
	  net.move('gpu') ;
	end

	if isfield(net,'normalization')
	  bopts = net.normalization;
	else
	  bopts = net.meta.normalization;
	end

	bopts.interpolation = 'bilinear';
	bopts.imageScales = opts.imageScales;
	bopts.numThreads = opts.numFetchThreads;
	bopts.useGpu = numel(opts.train.gpus) >  0 ;
	%Detect
	scores = cell(1,numel(testIdx));
	boxes = imdb.images.boxes(testIdx);
	names = imdb.images.name(testIdx);

	detLayer = find(arrayfun(@(a) strcmp(a.name, 'xTimes'), net.vars)==1);
	net.vars(detLayer(1)).precious = 1;
	% run detection
	start = tic ;
	for t=1:numel(testIdx)
	  batch = testIdx(t);  
	  
	  scoret = [];
	  for s=1:numel(opts.imageScales)
		for f=1:2 % add flips
		  inputs = getBatch(bopts, imdb, batch, opts.imageScales(s), f-1 );
		  net.eval(inputs) ;
	  
		  if isempty(scoret)
			scoret = squeeze(gather(net.vars(detLayer).value));
		  else
			scoret = scoret + squeeze(gather(net.vars(detLayer).value));
		  end
		end
	  end
	  scores{t} = scoret;
	  % show speed
	  time = toc(start) ;
	  n = t * 2 * numel(opts.imageScales) ; % number of images processed overall
	  speed = n/time ;
	  if mod(t,10)==0
		fprintf('test %d / %d speed %.1f Hz\n',t,numel(testIdx),speed);
	  end
	end

	dets.names  = names;
	dets.scores = scores;
	dets.boxes  = boxes;
	save(detsPath, '-struct','dets');
else
	dets = load(detsPath);
end
% --------------------------------------------------------------------
%                                                PASCAL VOC evaluation
% --------------------------------------------------------------------

if ~exist(apsPath)
    VOCinit;
    VOCopts.testset = testset;
    VOCopts.annopath = fullfile(opts.dataDir,'VOCdevkit','VOC2007','Annotations','%s.xml');
    VOCopts.imgsetpath = fullfile(opts.dataDir,'VOCdevkit','VOC2007','ImageSets','Main','%s.txt');
    VOCopts.localdir = fullfile(opts.dataDir,'VOCdevkit','local','VOC2007');
    cats = VOCopts.classes;
    ovTh = 0.4;
    scTh = 1e-3;
    aps = zeros(numel(cats),1);
    for cls = 1:numel(cats)
    
        vocDets.confidence = [];
        vocDets.bbox       = [];
        vocDets.ids        = [];
        
        for i=1:numel(dets.names)
            
            scores = double(dets.scores{i});
            boxes  = double(dets.boxes{i});
            
            boxesSc = [boxes,scores(cls,:)'];
            boxesSc = boxesSc(boxesSc(:,5)>scTh,:);
            pick = nms(boxesSc, ovTh);
            boxesSc = boxesSc(pick,:);
            
            vocDets.confidence = [vocDets.confidence;boxesSc(:,5)];
            vocDets.bbox = [vocDets.bbox;boxesSc(:,[2 1 4 3])];
            vocDets.ids = [vocDets.ids; repmat({dets.names{i}(1:6)},size(boxesSc,1),1)];
            
        end
        [rec,prec,ap] = wsddnVOCevaldet(VOCopts,cats{cls},vocDets,0);
        
        fprintf('%s %.1f\n',cats{cls},100*ap);
        aps(cls) = ap;
    end
    
    map = mean(aps);
    %write result to txt file
    fid = fopen(apsPath,'wt');
    fprintf(fid,'%s%f\n','map: ',map);
    fprintf(fid,'%f\n',aps);
    fclose(fid)

end


% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch, scale, flip)
% --------------------------------------------------------------------

opts.scale = scale;
opts.flip = flip;

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois] = wsddn_get_batch(images, imdb, batch, opts);


rois = single(rois');
if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
end

inputs = {'input', im, 'rois', rois} ;
  
  
if isfield(imdb.images,'boxScores')
  boxScore = reshape(imdb.images.boxScores{batch},[1 1 1 numel(imdb.images.boxScores{batch})]);
  inputs{end+1} = 'boxScore';
  inputs{end+1} = boxScore ; 
end

% -------------------------------------------------------------------------
function imdb = fixBBoxes(imdb, minSize, maxNum)

for i=1:numel(imdb.images.name)
  bbox = imdb.images.boxes{i};
  % remove small bbox
  isGood = (bbox(:,3)>=bbox(:,1)+minSize) & (bbox(:,4)>=bbox(:,2)+minSize);
  bbox = bbox(isGood,:);
  % remove duplicate ones
  [dummy, uniqueIdx] = unique(bbox, 'rows', 'first');
  uniqueIdx = sort(uniqueIdx);
  bbox = bbox(uniqueIdx,:);
  % limit number for training
  if imdb.images.set(i)~=3
    nB = min(size(bbox,1),maxNum);
  else
    nB = size(bbox,1);
  end
  
  if isfield(imdb.images,'boxScores')
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(isGood);
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(uniqueIdx);
    imdb.images.boxScores{i} = imdb.images.boxScores{i}(1:nB);
  end
  imdb.images.boxes{i} = bbox(1:nB,:);
  %   [h,w,~] = size(imdb.images.data{i});
  %   imdb.images.boxes{i} = [1 1 h w];
  
end


