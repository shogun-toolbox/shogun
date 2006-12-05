Creation and installation of Shogun-R
=====================================

- Enter R (this directory)
- type make sg_0.1-1.tar.gz
- in R/ there should be a new .tar.gz file now called sg_0.1-1.tar.gz
- either (as root) type make install R CMD INSTALL sg_0.1-1.tar.gz for 
 system wide installation or R CMD INSTALL --library=your_lib_path sg_0.1.tar.gz 
 for local installation
- don't forget to add your_lib_path to the R library path via adding the line
   
  .libPaths("your_lib_path")

  to your .Rprofile if you chose to do the later

- Now start R and type library() you should see shogun as the 
  package 'sg' with a small description.
