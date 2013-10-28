package regression;
use util;
use PDL;
use modshogun;

sub _evaluate (indata):
{
    my ($indata) = @_;
    my $prefix = 'kernel_';
    my $feats = &util::get_features($indata, $prefix);
    my $kargs = &util::get_args($indata, $prefix);
    my $fun = eval('modshogun::' . $indata->{$prefix.'name'}.'Kernel');
    my $kernel = $fun->new($feats->{'train'}, $feats->{'train'}, @$kargs);

    $prefix='regression_';
    $kernel->{parallel}->set_num_threads($indata->{$prefix.'num_threads'});
    my $rfun = eval('modshogun::' . $indata->{$prefix.'name'});
    if($@) {#except NameError, e:
	warn( "%s is disabled/unavailable!",$indata->{$prefix.'name'});
	return false;
    }
    my $labels = modshogun::RegressionLabels->new($indata->{$prefix.'labels'});
    if($indata->{$prefix.'type'} eq 'svm') {
	$regression = $rfun->new(
	    $indata->{$prefix.'C'}, $indata->{$prefix.'epsilon'}, $kernel, $labels);
    }elsif($indata->{$prefix.'type'} eq 'kernelmachine') {
	$regression = $rfun->new($indata->{$prefix.'tau'}, $kernel, $labels);
    }else{
	return false;
    }
    $regression->{parallel}->set_num_threads($indata->{$prefix.'num_threads'});
    if(defined($indata->{$prefix.'tube_epsilon'})) {
	$regression->set_tube_epsilon($indata->{$prefix.'tube_epsilon'});
    }
    $regression->train();

    my $alphas=0;
    my $bias=0;
    my $sv=0;
    if(defined($indata->{$prefix.'bias'})) {
	$bias = abs($regression->get_bias()-$indata->{$prefix.'bias'});
    }
    if(defined($indata->{$prefix.'alphas'})) {
	foreach my $item (@{ $regression->get_alphas()->tolist()}) {
	    $alphas+=$item;
	}
	$alphas=abs($alphas-$indata->{$prefix.'alphas'});
    }
    if(defined($indata->{$prefix.'support_vectors'})){
	foreach my $item (@{$inregression->get_support_vectors()->tolist()}) {
	    $sv+=$item;
	}
	$sv=abs($sv - $indata->{$prefix.'support_vectors'});
    }
    $kernel->init($feats->{'train'}, $feats->{'test'});
    my $classified=max(abs(
			   $regression->apply()->get_labels()-$indata->{$prefix.'classified'}));

    return &util::check_accuracy($indata->{$prefix.'accuracy'}
				 , {alphas=>$alphas,
				    bias=>$bias, support_vectors=>$sv, classified=>$classified});
}
########################################################################
# public
########################################################################
sub test
{
    my ($indata) = @_;
    return &_evaluate($indata);
}
1;
__END__


