function [ net ] = getFastRcnnModelZero(modelPath)
%Initicalize Fast Rcnn model
%   @G.Y.Guo

opts.piecewise = true;  % piecewise training (+bbox regression)
opts.modelPath = modelPath;

net = fast_rcnn_init( 'piecewise',opts.piecewise,...
  'modelPath',opts.modelPath);
end

