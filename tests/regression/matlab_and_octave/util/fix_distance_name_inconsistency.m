function y = fix_distance_name_inconsistency (dname)
	dname=upper(dname);
	if findstr('WORDDISTANCE', dname)
		pos=findstr('WORDDISTANCE', dname);
		y=dname(1:pos-1);
	elseif findstr('DISTANCE', dname)
		pos=findstr('DISTANCE', dname);
		y=dname(1:pos-1);
	elseif findstr('METRIC', dname)
		pos=findstr('METRIC', dname);
		y=dname(1:pos-1);
	else
		y=dname;
	end
