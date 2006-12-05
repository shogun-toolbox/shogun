function s = calcrocscore(oute,LTE)
% rocscore = calcrocscore(output,LT)
%
% computes the area under the ROC curve

[a,b]=calcroc(oute,LTE) ;

s=0 ;
for i=1:length(a)-1,
  s = s + (a(i)-a(i+1))*(b(i)+b(i+1))/2 ;
end ;

