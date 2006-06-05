Creation and installation of Shogun-R
=====================================

- Enter R.pack (this directory)
- type make
- in R.pack/ there should be a new .tar.gz file now called sg_0.1.tar.gz
- type R CMD INSTALL --library=your_lib_path sg_0.1.tar.gz
- Add your_lib_path to the R library path via adding the line
   
  .libPaths("your_lib_path")

  to your .Rprofile

- Now start R and type library() you should see shogun as the 
  package 'sg' with a small description.
