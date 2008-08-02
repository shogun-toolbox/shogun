tobool <- function(value) {
	evaluated=eval(parse(text=toupper(value)))
	if (typeof(evaluated)=='logical') {
		return(evaluated)
	} else {
		print(paste('Could not make a bool out of', value))
		return(FALSE)
	}
}
