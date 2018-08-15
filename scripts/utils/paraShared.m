function [net] = paraShared( net1,net2)
%copy net2 parameter to net1
%   author @ggy

net = net1;
for i = 1:26
    net.params(i).value = net2.params(i).value;
    net.params(i).learningRate = 0.1;
end



