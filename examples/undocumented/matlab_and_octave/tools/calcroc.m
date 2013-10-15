function [false_alarms,hits] = calcroc(output,LTE)

assert(all(size(output)==size(LTE))) ;

[ld,idx] = sort(output);

hits=1-cumsum(LTE(idx)>0)/sum(LTE > 0) ;
false_alarms=1-cumsum(LTE(idx)<0)/sum(LTE < 0) ;

hits = [1 hits ];
false_alarms = [1 false_alarms];

if nargout<2,
  false_alarms = [false_alarms;hits];
end ;
