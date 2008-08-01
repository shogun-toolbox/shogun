set_kernel <- function() {
	kname <- name
	ftype <- toupper(feature_type)
	size_cache <- 10

	sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width)
}
