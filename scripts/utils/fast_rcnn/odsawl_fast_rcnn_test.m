function [detsTrainval] = odsawl_fast_rcnn_test(gpus,...
	datasetPath,expPath,imdb,...
	varargin)
%test script for FastRcnn in ODSAW
%	@author: G.Y. Guo

opts.gpu = gpus ;
opts.numFetchThreads = 1 ;
opts.nmsThresh = 0.3 ;
opts.maxPerImage = 100 ;
opts = vl_argparse(opts, varargin) ;

opts.expDir = expPath;
opts.dataDir = datasetPath;
opts.modelPath = fullfile(opts.expDir, 'net-deployed.mat') ;
display(opts) ;

addpath(fullfile(opts.dataDir, 'VOCdevkit', 'VOCcode'));

if ~exist(opts.expDir,'dir')
  error('no folder %s exist',opts.expDir) ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------


net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
net.mode = 'test' ;
% -------------------------------------------------------------------------
%                                                            Detect
% -------------------------------------------------------------------------
detsOnTestPath = fullfile(expPath,'detsTest.mat');
detsOnTrainvalPath = fullfile(expPath,'detsTrainval.mat');
apsOnTestPath = fullfile(expPath,'apsTest.txt');
apsOnTrainvalPath = fullfile(expPath,'apsTrainval.txt');


[detsTest] = test(opts, net, imdb, 'test',detsOnTestPath,apsOnTestPath);
fprintf('fast_rcnn test on test done\n');

[detsTrainval] = test(opts, net, imdb, 'trainval',detsOnTrainvalPath,apsOnTrainvalPath);
fprintf('fast_rcnn test on trainval done\n');



% --------------------------------------------------------------------
function [boxscores_nms ] = test(opts, net, imdb, testset,detsPath,apsPath)
% --------------------------------------------------------------------

if ~isempty(opts.gpu)
  net.move('gpu') ;
end

%set test 
if strcmp(testset, 'test')
	testIdx = find(imdb.images.set == 3);
elseif strcmp(testset, 'trainval')
	imdb.images.set(imdb.images.set == 2) = 1;
	testIdx = find(imdb.boxes.flip == 0 & imdb.images.set == 1);
else
	error('no testst %s', testset);
end

VOCinit;
VOCopts.testset=testset;


if ~exist(detsPath)

	bopts.averageImage = net.meta.normalization.averageImage;
	bopts.useGpu = numel(opts.gpu) >  0 ;
	bopts.maxScale = 1000;
	bopts.bgLabel = 21;
	bopts.visualize = 0;
	bopts.scale = 600;
	bopts.interpolation = net.meta.normalization.interpolation;
	bopts.numThreads = opts.numFetchThreads;

	cls_probs  = cell(1,numel(testIdx)) ;
	box_deltas = cell(1,numel(testIdx)) ;
	boxscores_nms = cell(numel(VOCopts.classes),numel(testIdx)) ;
	ids = cell(numel(VOCopts.classes),numel(testIdx)) ;

	dataVar = 'input' ;
	probVarI = net.getVarIndex('probcls') ;
	boxVarI = net.getVarIndex('predbbox') ;
	if isnan(probVarI)
	  dataVar = 'data' ;
	  probVarI = net.getVarIndex('cls_prob') ;
	  boxVarI = net.getVarIndex('bbox_pred') ;

	end

	net.vars(probVarI).precious = true ;
	net.vars(boxVarI).precious = true ;
	start = tic ;
	for t=1:numel(testIdx)
	  speed = t/toc(start) ;
	  fprintf('Image %d of %d (%.f HZ)\n', t, numel(testIdx), speed) ;
	  batch = testIdx(t);
	  inputs = getBatch(bopts, imdb, batch);
	  inputs{1} = dataVar ;
	  net.eval(inputs) ;

	  cls_probs{t} = squeeze(gather(net.vars(probVarI).value)) ;
	  box_deltas{t} = squeeze(gather(net.vars(boxVarI).value)) ;
    end
    
    

	% heuristic: keep an average of 40 detections per class per images prior
	% to NMS
	max_per_set = 40 * numel(testIdx);

	% detection thresold for each class (this is adaptively set based on the
	% max_per_set constraint)
	cls_thresholds = zeros(1,numel(VOCopts.classes));
	cls_probs_concat = horzcat(cls_probs{:});


	for c = 1:numel(VOCopts.classes)
	  q = find(strcmp(VOCopts.classes{c}, net.meta.classes.name)) ;
	  so = sort(cls_probs_concat(q,:),'descend');
	  cls_thresholds(q) = so(min(max_per_set,numel(so)));
	  fprintf('Applying NMS for %s\n',VOCopts.classes{c});

	  for t=1:numel(testIdx)
		si = find(cls_probs{t}(q,:) >= cls_thresholds(q)) ;
		if isempty(si), continue; end
		cls_prob = cls_probs{t}(q,si)';
		pbox = imdb.boxes.pbox{testIdx(t)}(si,:);

		% back-transform bounding box corrections
		delta = box_deltas{t}(4*(q-1)+1:4*q,si)';
		pred_box = bbox_transform_inv(pbox, delta);

		im_size = imdb.images.size(testIdx(t),[2 1]);
		pred_box = bbox_clip(round(pred_box), im_size);

		% Threshold. Heuristic: keep at most 100 detection per class per image
		% prior to NMS.
		boxscore = [pred_box cls_prob];
		[~,si] = sort(boxscore(:,5),'descend');
		boxscore = boxscore(si,:);
		boxscore = boxscore(1:min(size(boxscore,1),opts.maxPerImage),:);

		% NMS
		pick = bbox_nms(double(boxscore),opts.nmsThresh);

		boxscores_nms{c,t} = boxscore(pick,:) ;
		ids{c,t} = repmat({imdb.images.name{testIdx(t)}(1:end-4)},numel(pick),1) ;

		if 0
		  figure(1) ; clf ;
		  idx = boxscores_nms{c,t}(:,5)>0.5;
		  if sum(idx)==0, continue; end
		  bbox_draw(imread(fullfile(imdb.imageDir,imdb.images.name{testIdx(t)})), ...
					boxscores_nms{c,t}(idx,:)) ;
		  title(net.meta.classes.name{q}) ;
		  drawnow ;
		  pause;
		  %keyboard
		end
	  end
	end
	
	save(detsPath,'boxscores_nms','cls_probs','cls_thresholds','box_deltas','-v7.3');
	
else
	boxscores_nms = load(detsPath);
	boxscores_nms = boxscores_nms.boxscores_nms;
end

if ~exist(apsPath)
	%% PASCAL VOC evaluation
	VOCdevkitPath = fullfile(opts.dataDir,'VOCdevkit');
	aps = zeros(numel(VOCopts.classes),1);

	% fix voc folders
	VOCopts.imgsetpath = fullfile(VOCdevkitPath,'VOC2007','ImageSets','Main','%s.txt');
	VOCopts.annopath   = fullfile(VOCdevkitPath,'VOC2007','Annotations','%s.xml');
	VOCopts.localdir   = fullfile(VOCdevkitPath,'local','VOC2007');
	VOCopts.detrespath = fullfile(VOCdevkitPath, 'results', 'VOC2007', 'Main', ['%s_det_', VOCopts.testset, '_%s.txt']);

	% write det results to txt files
	for c=1:numel(VOCopts.classes)
	  fid = fopen(sprintf(VOCopts.detrespath,'comp3',VOCopts.classes{c}),'w');
	  for i=1:numel(testIdx)
		if isempty(boxscores_nms{c,i}), continue; end
		dets = boxscores_nms{c,i};
		for j=1:size(dets,1)
		  fprintf(fid,'%s %.6f %d %d %d %d\n', ...
			imdb.images.name{testIdx(i)}(1:end-4), ...
			dets(j,5),dets(j,1:4)) ;
		end
	  end
	  fclose(fid);
	  [rec,prec,ap] = VOCevaldet(VOCopts,'comp3',VOCopts.classes{c},0);
	  fprintf('%s ap %.1f\n',VOCopts.classes{c},100*ap);
	  aps(c) = ap;
	end
	fprintf('mean ap %.1f\n',100*mean(aps));
	
	map = mean(aps);
    %write result to txt file
    fid = fopen(apsPath,'wt');
    fprintf(fid,'%s%f\n','map: ',map);
    fprintf(fid,'%f\n',aps);
    fclose(fid)
end	
	
% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
if isempty(batch)
  return;
end

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois] = fast_rcnn_eval_get_batch(images, imdb, batch, opts);

rois = single(rois);
if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
end

inputs = {'input', im, 'rois', rois} ;