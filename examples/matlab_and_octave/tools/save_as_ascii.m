function save_as_ascii(XT, fname) ;
% save_as_ascii(XT, fname) ;

fd = fopen(fname, 'w+') ;
for i=1:size(XT,2) ;
  fprintf(fd, '%s\n', XT(:,i)) ;
end ;
fclose(fd) ;

