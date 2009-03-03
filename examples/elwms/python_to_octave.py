from elwms import elwms

elwms('loglevel', 'ALL')
elwms('run_octave','octavecode', 'disp("hi")')
x,y=elwms('run_octave','A', ['test','bla','foo'], 'octavecode', 'disp(A); disp("hi"); results=list({"a","b","c"},{"c","d"})')
