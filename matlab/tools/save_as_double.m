function save_as_double(L, fname) ;
% save_as_ascii(L, fname) ;

fd = fopen(fname, 'w+') ;
for i=1:size(L,2) ;
  fwrite(fd, L(:,i), 'double') ;
end ;
fclose(fd) ;

