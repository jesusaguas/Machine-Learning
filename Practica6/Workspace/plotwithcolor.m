function plotwithcolor(x, label)
% plots 2D points with different color according to label


colorMap =[1 1 0;
1 0 1;
0 1 1;
1 0 0;
0 1 0;
0 0 1;
1 1 1;
.3 .6 .1;
.1 .7 .2;
.3 .3 .3];

values=unique(label);

legendstr={};
for I=1:length(values),
	ind=find(label==values(I)); 
    hh=plot(x(ind,1), x(ind,2),'.');
    set(hh, 'Color',colorMap(I,:));
    legendstr{I}=int2str(values(I));
end

legend(legendstr)