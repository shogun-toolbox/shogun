
read_mfile <- function(filename){
	f <- file(filename)
	lines <- readLines(f)
	for (line in lines){
		if(regexpr("[[]",line)>0){
			#return(line)
			n <- nchar(line)
			line <- gsub(";","::",line)
			rows <- nchar(line)-n+1
			line <- gsub("::",",",line)
			line <- gsub("=", "<-",line)
			line <- gsub("[[]","matrix(c(",line)
			line <- gsub("[]]",paste("),nrow=",rows, ")"),line)
			line <- gsub(";",",",line)
			#return(line)
			tcon <- textConnection(line)
			source(tcon)
			close(tcon)
		}else {
			line <- gsub("=","<- ",line)
			tcon <- textConnection(line)
			source(tcon)
			close(tcon)
		}
	}
}
