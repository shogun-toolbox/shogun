cd ~/
fd=fopen('Z.dat', 'r') ; z=fread(fd,inf,'double') ; fclose(fd) ;
fd=fopen('A.dat', 'r') ; a=fread(fd,inf,'double') ; fclose(fd) ;
fd=fopen('b.dat', 'r') ; b=fread(fd,inf,'double') ; fclose(fd) ;
fd=fopen('u.dat', 'r') ; u=fread(fd,inf,'double') ; fclose(fd) ;
fd=fopen('l.dat', 'r') ; l=fread(fd,inf,'double') ; fclose(fd) ;
fd=fopen('r.dat', 'r') ; r=fread(fd,inf,'double') ; fclose(fd) ;
fd=fopen('c.dat', 'r') ; c=fread(fd,inf,'double') ; fclose(fd) ;

Z=reshape(z,3,5000) ;
