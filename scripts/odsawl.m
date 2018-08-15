function odsawl(varargin)
%train and test code for ODSAWL
%	@author: G.Y. Guo
clearvars -except varargin
clc;

run('../matconvnet/matlab/vl_setupnn.m') 

warning off
opts.gpus =[8]; %gpu device ,start from 1
opts.proposalType = 'EB'; %region proposal type, 'SSW' or 'EB'
opts.labelNumPerCls = 50;%number of labeled images
opts.iteNum = 4;  % number of iterations

%% set path
opts.sswproposalPath = '../data/SSW';
opts.ebproposalPath = '../data/EB';
opts.datasetPath = '../data/VOC2007';
opts.modelPath = '../models/imagenet-vgg-verydeep-16.txt';
opts.dataPath = fullfile('../data',['ODSAWL_' num2str(opts.labelNumPerCls) '_' opts.proposalType]);
opts.trainDataPath = fullfile(opts.dataPath,'trainData.mat');
opts.imageStatsPath = fullfile(opts.dataPath, 'imageStats.mat');
opts.trainDataPath = fullfile(opts.dataPath,'trainData.mat');
opts.imdbwsddnPath = fullfile(opts.dataPath,'imdbwsddn.mat');
opts.imdbFastRcnnPath = fullfile(opts.dataPath,'imdbFastRcnn.mat');
opts.netwsddnZeroPath = fullfile(opts.dataPath,'netwsddn.mat');
opts.netFastRcnnZeroPath = fullfile(opts.dataPath,'netFastRcnn.mat');
opts.diaryPath = fullfile(opts.dataPath,'diary');%path for Log

opts = vl_argparse(opts, varargin) ;

addpath(genpath('./utils'));
addpath('../matconvnet/examples/imagenet');
addpath('../matconvnet/examples');
addpath(genpath('../matconvnet/examples/fast_rcnn'));
addpath(genpath('../matconvnet/contrib/WSDDN'));
run('../matconvnet/matlab/vl_setupnn.m')

%% set up data path and diary path
if ~exist(opts.dataPath)
    mkdir(opts.dataPath);
end
if ~exist(opts.diaryPath)
    mkdir(opts.diaryPath);
end
diaryName = fullfile(opts.diaryPath,['diary-' strrep(num2str(fix(clock)),' ','') '.txt']);
diary(diaryName);

%%  set imdb of wsddn and fastrcnn
%generate imdb of wsddn
fprintf('loading imdb for WSDDN...');
if ~exist(opts.imdbwsddnPath)
    if strcmp(opts.proposalType, 'SSW')
        imdbwsddn = setup_voc07_eb('dataDir',opts.datasetPath, ...
            'proposalDir',opts.sswproposalPath,'loadTest',1);
    else
        imdbwsddn = setup_voc07_eb('dataDir',opts.datasetPath, ...
            'proposalDir',opts.ebproposalPath,'loadTest',1);
    end
    save(opts.imdbwsddnPath,'-struct', 'imdbwsddn', '-v7.3');
else
    imdbwsddn = load(opts.imdbwsddnPath);
end
fprintf('done\n');

%generate/load imdb of FastRcnn
fprintf('loading imdb for FastRcnn...\n');
if ~exist(opts.imdbFastRcnnPath)
    if strcmp(opts.proposalType, 'SSW')
        imdbFastRcnn = cnn_setup_data_voc07_ssw(...
            'dataDir', opts.datasetPath, 'sswDir',opts.sswproposalPath,...
            'addFlipped', true, 'useDifficult', true) ;
    else
        imdbFastRcnn = cnn_setup_data_voc07_eb(...
            'dataDir', opts.datasetPath,'ebDir', opts.ebproposalPath, ...
            'addFlipped', true,'useDifficult', true) ;   
    end
    save(opts.imdbFastRcnnPath,'-struct', 'imdbFastRcnn','-v7.3');
else
    imdbFastRcnn  = load(opts.imdbFastRcnnPath);
end
fprintf('done\n');

%% get model zero
fprintf('load net zero for WSDDN...\n');
if ~exist(opts.netwsddnZeroPath)
    netwsddnZero = getwsddnModelZero(imdbwsddn,opts);
    netwsddnZero_ = netwsddnZero;
    netwsddnZero_ = netwsddnZero_.saveobj() ;
    save(opts.netwsddnZeroPath, '-struct','netwsddnZero_');
else
    netwsddnZero = load(opts.netwsddnZeroPath);
end
fprintf('done\n');

fprintf('load net zero for FastRcnn...\n');
if ~exist(opts.netFastRcnnZeroPath)
    netFastRcnnZero = getFastRcnnModelZero(opts.modelPath);
    netFastRcnnZero_ = netFastRcnnZero;
    netFastRcnnZero_ = netFastRcnnZero_.saveobj() ;
    save(opts.netFastRcnnZeroPath, '-struct','netFastRcnnZero_');
else
    netFastRcnnZero = load(opts.netFastRcnnZeroPath)
end
fprintf('load net zero for FastRcnn done\n');


%% chose  images  which have label
if ~exist(opts.trainDataPath)
    
    trainData = get_traindata(imdbwsddn,opts.labelNumPerCls);
    save(opts.trainDataPath, 'trainData');
else
    trainData = load(opts.trainDataPath);
    trainData = trainData.trainData;
end


for ite = 1:opts.iteNum
    fprintf('************** iteration: %d ****************\n', ite);
    %set path
    curDataPath = fullfile(opts.dataPath,['round' num2str(ite)]);
    curwsddnPath=fullfile(curDataPath,'WSDDN');
    curFastRcnnPath=fullfile(curDataPath,'FastRcnn');
    if ~exist(curDataPath)
        mkdir(curDataPath);
    end
    if ~exist(curwsddnPath)
        mkdir(curwsddnPath);
    end
    if ~exist(curFastRcnnPath)
        mkdir(curFastRcnnPath);
    end
    
    %% -------------------------------------WSDDN---------------------------
    if ite == 1
        curwsddnNet = dagnn.DagNN.loadobj(netwsddnZero);
        curwsddnImdb = imdbwsddn;
        trainIdx = trainData;
    else
        %load the net of previous iteration
        lastFastRcnnNet = load(fullfile(opts.dataPath,['round' num2str(ite-1)],...
            'FastRcnn','net-epoch-12.mat'));
        lastFastRcnnNet =  lastFastRcnnNet.net;
        curwsddnNet = paraShared(netwsddnZero,lastFastRcnnNet);
        curwsddnNet= dagnn.DagNN.loadobj(curwsddnNet) ;
        %load detect results of FastRcnn in last iteration
        detsFastRcnnTrainval = load(fullfile(opts.dataPath,['round' num2str(ite-1)],...
            'FastRcnn','detsTrainval.mat'));
        detsFastRcnnTest = load(fullfile(opts.dataPath,['round' num2str(ite-1)],'FastRcnn','detsTest.mat'));
        
        %generate wsddn imdb file for current iteration
        [curtrainData,curwsddnImdb] =  getwsddnImdb(detsFastRcnnTrainval,...
            detsFastRcnnTest,imdbwsddn,trainData,500*(ite-1));
        
        trainIdx = curtrainData;
    end
    save(fullfile(curwsddnPath,['imdbwsddn' num2str(ite) '.mat']),...
        '-struct', 'curwsddnImdb', '-v7.3');
    
    %train wsddn
    if ~exist(fullfile(curwsddnPath,'net-epoch-20.mat'))
        [net, info] = odsawl_wsddn_train(curwsddnPath,curwsddnNet,...
            curwsddnImdb,trainIdx,1,opts.gpus);
    end
    fprintf('train WSDDN for iteration: %d done\n', ite);
    
    %test and get results
    detswsddnTrainval = odsawl_wsddn_test(opts.datasetPath,curwsddnPath,imdbwsddn,opts.gpus);
    restrainvalwsddn  = getReswsddn( detswsddnTrainval,trainData,imdbwsddn,0.4,1e-3 );
    fprintf('test WSDDN for iteration: %d done\n', ite);
    
    %% ----------------------------Fast Rcnn----------------------------------
    %get imdb file for current iteration FastRcnn train
    if ~exist(fullfile(curFastRcnnPath,['imdbFastRcnn' num2str(ite) '.mat']))
        if strcmp(opts.proposalType, 'SSW')
            curFastRcnnImdb = odsawl_setup_data_voc07_ssw_addbox(restrainvalwsddn,...
                imdbwsddn,imdbFastRcnn,'dataDir', opts.datasetPath, ...
                'sswDir', opts.sswproposalPath,'addFlipped', true,'useDifficult', true) ;
        else
            curFastRcnnImdb = odsawl_setup_data_voc07_eb_addbox(restrainvalwsddn,...
                imdbwsddn,imdbFastRcnn, 'dataDir', opts.datasetPath, ...
                'ebDir', opts.ebproposalPath, 'addFlipped', true,  'useDifficult', true) ;
        end
        save(fullfile(curFastRcnnPath,['imdbFastRcnn' num2str(ite) '.mat']),...
            '-struct', 'curFastRcnnImdb', '-v7.3');
    else
        curFastRcnnImdb = load(fullfile(curFastRcnnPath,['imdbFastRcnn' num2str(ite) '.mat']));
    end
    
    %train FastRcnn
    if ~exist(fullfile(curFastRcnnPath,'net-deployed.mat'))
        lastwsddnNet = load(fullfile(opts.dataPath,['round' num2str(ite)],'WSDDN','net-epoch-20.mat'));
        lastwsddnNet =  lastwsddnNet.net;
        curFastRcnnNet = paraShared(netFastRcnnZero,lastwsddnNet);
        curFastRcnnNet  = dagnn.DagNN.loadobj(curFastRcnnNet );
        [net, info] = odsawl_fast_rcnn_train_fine_tuning(opts.gpus,...
            curFastRcnnPath,curFastRcnnNet,curFastRcnnImdb,1);
    end
    fprintf('train FastRcnn for iteration: %d done\n', ite);
    
    %test current FastRcnn net and return results on train and val
    detsFastRcnnTrainval = odsawl_fast_rcnn_test(opts.gpus,...
        opts.datasetPath,curFastRcnnPath,imdbFastRcnn);
	fprintf('test FastRcnn for iteration: %d done\n', ite);
    
end

end

