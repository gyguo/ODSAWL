function [newTrainData,imdb ] = getwsddnImdb( detstrainval,detstest,imdbwsddn,trainData,addNum)
%Get imdb for wsddn train of next iteration
%   author @G.Y.Guo

imdb = imdbwsddn;
newTrainData = trainData;
imdbwsddn.images.set(imdbwsddn.images.set == 2) = 1;
trainIdx = find(imdbwsddn.images.set == 1);
testIdx = find(imdbwsddn.images.set == 3);

%get index of label data and unlabeled data
untrainData = setdiff(trainIdx,trainData);
[~,trainDataIdx,~] = intersect(trainIdx , trainData);
[~,untrainDataIdx,~] = intersect(trainIdx , untrainData);

imgScore = zeros(numel(untrainData),1);
resCls = detstrainval.cls_probs;
for i = 1:numel(untrainData)

    curCls = resCls{untrainDataIdx(i)};
    curscore = max(curCls,[],2);
    curscore(21) = mean(curCls(21,:));
    
    curLabelscores = [];
    for cls = 1:20
        if curscore(cls)>=curscore(21)
             curLabelscores =  [ curLabelscores;curscore(cls)];
        end
    end
    if ~isempty(curLabelscores)
        imgScore(i) = mean(curLabelscores)*exp(-1*numel(curLabelscores));
    end
end

[imgScore,imgIdx] = sort(imgScore,'descend');

for i =1:min(addNum,numel(find(imgScore>0)))
    curIdx = untrainDataIdx(imgIdx(i));
    curImg = trainIdx(curIdx);
	
    newTrainData = horzcat(newTrainData,curImg);
	imdb.images.label(:,curImg)=-1;
	
    curCls = resCls{curIdx};
    curscore = max(curCls,[],2);
    curscore(21) = mean(curCls(21,:));
	
    curLabes = [];
    for cls = 1:20
        if curscore(cls)>=curscore(21)
             curLabes =  [ curLabes;cls];
			 imdb.images.label(cls,curImg)=1;
        end
    end
	if isempty(curLabes)
        error('image without positive labels should not be chosed');
    end
end
newTrainData = unique(newTrainData );
