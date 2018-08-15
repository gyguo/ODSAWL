function [trainData] = get_traindata(imdbwsddn,labelNumPerCls)
%select labeled data for train
%author @G.Y.Guo

 % use train + val for training
    imdbwsddn.images.set(imdbwsddn.images.set == 2) = 1;
    trainIdx = find(imdbwsddn.images.set == 1);
numPos([1:20],1)=0;
for i=1:numel(trainIdx)
    for j=1:20
        if imdbwsddn.images.label(j,trainIdx(i))==1
            numPos(j)=numPos(j)+1;
            posImgs(j,numPos(j))=trainIdx(i);
        end
    end
end

numNeg([1:20],1)=0;
for i=1:numel(trainIdx)
    for j=1:20
        if imdbwsddn.images.label(j,trainIdx(i))==-1
            numNeg(j)=numNeg(j)+1;
            negImgs(j,numNeg(j))=trainIdx(i);
        end
    end
end

%chose positive images per class
trainNum=0;
for i=1:20
    curPosnum=numPos(i);
    curPos=posImgs(i,[1:curPosnum]);
    for j=1:labelNumPerCls
        trainNum=trainNum+1;
        trainData(trainNum)=randsrc(1,1,curPos);
        curPos(curPos==trainData(trainNum))=[];
    end
end

%chose  negative images per class
for i=1:20
    curNegnum=numNeg(i);
    curNeg=negImgs(i,[1:curNegnum]);
    for j=1:labelNumPerCls
        trainNum=trainNum+1;
        trainData(trainNum)=randsrc(1,1,curNeg);
        curNeg(curNeg==trainData(trainNum))=[];
    end
end

trainData=sort(trainData);
trainData=unique(trainData);

end

