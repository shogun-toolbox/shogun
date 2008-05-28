

function y = test_octave(filename)
  
  pos = max(findstr('/', filename));
  res = test_kernels(filename(pos+1:end));

  if(res == 0)
    sprintf('__OK__')
  else
    sprintf( '__ERR__')
  end



