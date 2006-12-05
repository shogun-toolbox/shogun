# By convention the file zzz.R in an R package
# contains two routinges called .First.lib and
# .Last.lib which are called when the package 
# is loaded respective unloaded during a R
# session.

# Load the shogun dynamic library at startup.
#
.First.lib <- function(lib,pkg) library.dynam("sg",pkg,lib) #dyn.load("../libs/sg.so")  

# Unload the library.
#
.Last.lib <- function() dyn.unload("sg.so") 

# Because in packages with namespaces .First.lib will not be loaded
# one needs another functions called .onLoad resp. .onUnload
#
.onLoad <- function() .First.lib()
.onUnload <- function() .Last.lib()

# a nice banner
#
.onAttach <- function() cat(paste("\nWelcome! This is ShoGun version 0.1\n"))
