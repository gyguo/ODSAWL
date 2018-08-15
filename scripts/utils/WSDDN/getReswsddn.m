function [ trainvalRes ] = getReswsddn( dets,trainData,imdbwsddn,ovTh,scTh )
%Get wsddn test results on trainval set
%   author @G.Y.Guo

categories= {'aeroplane','bicycle','bird','boat','bottle','bus','car',...
  'cat','chair','cow','diningtable','dog','horse','motorbike','person',...
  'pottedplant','sheep','sofa','train','tvmonitor'};

imdbwsddn.images.set(imdbwsddn.images.set == 2) = 1;
trainIdx = find(imdbwsddn.images.set == 1);
assert(numel(dets.names)==numel(trainIdx));

trainvalRes ={};
for i=1:numel(dets.names)
    
    vocDets.confidence = [];
    vocDets.bbox       = [];
    vocDets.cats       = [];
	vocDets.istrain      = false;
    
    scores = double(dets.scores{i});
    boxes  = double(dets.boxes{i});
    
    %modify the trainData by its class
    isTrainData = 0;
    curCls  = [];
    if find(trainData==trainIdx(i))
        isTrainData = 1;
		vocDets.istrain  = true;
        curCls = find(imdbwsddn.images.label(:,trainIdx(i))==1) ; 
    end
    
    for cls = 1:numel(categories)
        if isTrainData==0 | numel(find(curCls==cls)) ~= 0
            
            boxesSc = [boxes,scores(cls,:)'];
            boxesSc = boxesSc(boxesSc(:,5)>scTh,:);
            pick = nms(boxesSc, ovTh);
            boxesSc = boxesSc(pick,:);
            
            if  numel(boxesSc)~=0
                vocDets.confidence = [vocDets.confidence;boxesSc(:,5)];
                vocDets.bbox = [vocDets.bbox;boxesSc(:,[2 1 4 3])];
                vocDets.cats = [vocDets.cats;repmat(cls,size(boxesSc,1),1)];
            end
        end
    end
    trainvalRes = [trainvalRes;vocDets];
end





