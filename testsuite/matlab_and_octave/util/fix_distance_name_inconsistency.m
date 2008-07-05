function y = fix_distance_name_inconsistency (dname)
	dname=toupper(dname);
	if findstr('WORDDISTANCE', dname)
		pos=findstr('WORDDISTANCE', dname);
		y=dname(1:pos);
	elseif findstr('DISTANCE', dname)
		pos=findstr('DISTANCE', dname);
		y=dname(1:pos);
	elseif findstr('METRIC', dname)
		pos=findstr('METRIC', dname);
		y=dname(1:pos);
	else:
		y=dname;
	end
