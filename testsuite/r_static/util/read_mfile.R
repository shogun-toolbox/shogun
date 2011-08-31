read_mfile <- function(filename) {
	res <- c()
	f <- file(filename)
	lines <- readLines(f)

	for (line in lines) {
		if (regexpr("[[]",line)>0) {
			n <- nchar(line)
			line <- gsub(";","::",line)
			rows <- nchar(line)-n
			#rows <- nchar(line)-n+1
			line <- gsub("::",",",line)
			line <- gsub("=", "<-",line)
			line <- gsub("[[]","matrix(c(",line)
			line <- gsub("[]]",paste("),nrow=",rows, ",byrow=TRUE)"),line)
			line <- gsub(";",",",line)
			line <- gsub(",$", "", line) # remove trailing comma
			res <- cbind(res, line)
			#tcon <- textConnection(line)
			#source(tcon)
			#close(tcon)
		} else {
			line <- gsub("=","<- ",line)
			line <- gsub("[{]", "c(",line)
			line <- gsub("[}]", ')', line)
			res <- cbind(res, line)
			#tcon <- textConnection(line)
			#source(tcon)
			#close(tcon)
		}
	}

	close(f)
	return(res)
}
