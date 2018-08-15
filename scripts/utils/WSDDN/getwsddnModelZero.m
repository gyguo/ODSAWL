function [ net ] = getwsddnModelZero(imdb,opts)
%Get the model for WSDDN without train
%   author @G.Y.Guo
%% Compute image statistics (mean, RGB covariances, etc.)
images = imdb.images.name(imdb.images.set == 1) ;
images = strcat([imdb.imageDir filesep],images) ;

if exist(opts.imageStatsPath,'file')
    load(opts.imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
    [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
        'imageSize', [256 256], ...
        'numThreads', 1, ...
        'gpus', opts.gpus) ;
    save(opts.imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

nopts.averageImage = reshape(rgbMean,[1 1 3]) ;
% nopts.rgbVariance = 0.1 * rgbDeviation ;
nopts.rgbVariance = [] ;
nopts.numClasses = numel(imdb.classes.name) ;
nopts.classNames = imdb.classes.name ;
nopts.addBiasSamples = 1; % add Box Scores (only with Edge Boxes)
nopts.addLossSmooth  = 1; % add Spatial Regulariser
nopts.softmaxTempCls = 1; % softmax temp for cls
nopts.softmaxTempDet = 2; % softmax temp for det

net = importdata(opts.modelPath);
net = wsddn_init(net,nopts);

end

