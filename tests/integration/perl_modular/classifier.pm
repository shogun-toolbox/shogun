package classifier;
use modshogun;
use util;
use PDL;
use Data::Dumper;

sub _get_machine
{
    my  ($indata, $prefix, $feats) = @_;
    my $machine;
    if ($indata->{$prefix.'type'} eq 'kernel') {
	my $pre='kernel_';
	my $kargs=&util::get_args($indata, $pre);
	my $kfun=*{$indata->{$pre.'name'}.'Kernel'};
	my $machine=$kfun->($feats->{'train'}, $feats->{'train'}, $kargs);
	if ($indata->{$pre.'name'} eq 'Linear') {
	    my $normalizer=*{$indata->{$pre.'normalizer'}->()};
	    $machine->set_normalizer($normalizer);
	    $machine->init($feats->{'train'}, $feats->{'train'});
	}
	$machine->parallel->set_num_threads($indata->{$prefix.'num_threads'});
    } elsif( $indata->{$prefix.'type'} eq 'knn') {
	my $pre='distance_';
	my $dargs=&util::get_args($indata, $pre);
	my $dfun= *{$indata->{$pre.'name'}};
	$machine=$dfun->($feats->{'train'}, $feats->{'train'}, $dargs);
	$machine->parallel->set_num_threads($indata->{$prefix.'num_threads'});
    } else {
	$machine=undef;
    }
    return $machine;
}
sub  _get_results_alpha_and_sv
{
    my ($indata, $prefix, $classifier) = @_;
    if( not defined($indata->{$prefix.'alpha_sum'})
	and
	not defined($indata->{$prefix.'sv_sum'})){
	return (undef, undef);
    }
    my $a=0;
    my $sv=0;
    if( defined($indata->{$prefix.'label_type'})
	and
	$indata->{$prefix.'label_type'} eq 'series'){
	foreach my  $i (0..$classifier->get_num_svms()){
	    my $subsvm=$classifier->get_svm($i);
	    foreach my  $item (@{$subsvm->get_alphas()}){
		$a+=$item;
	    }
	    foreach my $item (@{$subsvm->get_support_vectors()}){
		$sv+=$item;
	    }
	}
	$a=abs($a-$indata->{$prefix.'alpha_sum'});
	$sv=abs($sv-$indata->{$prefix.'sv_sum'});
    }else{
	foreach my $item (@{$classifier->get_alphas()}){
	    $a+=$item;
	    $a=abs($a-$indata->{$prefix.'alpha_sum'});
	    foreach my $item (@{$classifier->get_support_vectors()->tolist()}){
		$sv+=$item;
	    }
	    $sv=abs($sv-$indata->{$prefix.'sv_sum'});
	}
    }
    return ($a, $sv);
}

sub _get_results
{
    my ($indata, $prefix, $classifier, $machine, $feats) = @_;
    my $e;
    my %res=(
		'alphas'=>0,
		'bias'=>0,
		'sv'=>0,
		'accuracy'=>$indata->{$prefix.'accuracy'}
	);

    if(defined($indata->{$prefix.'bias'})) {
	    $res{'bias'}=abs($classifier->get_bias()-$indata->{$prefix.'bias'});
    }
    ($res{'alphas'}, $res{'sv'})
	= &_get_results_alpha_and_sv($indata, $prefix, $classifier);

    my $ctype=$indata->{$prefix.'type'};
    if( $ctype eq 'kernel' or $ctype eq 'knn' ) {
	$machine->init($feats->{'train'}, $feats->{'test'});
    }else{
	@ctypes=('linear', 'perceptron', 'lda', 'wdsvmocas');
	if(grep($ctype, @ctypes)) {
	    $classifier->set_features($feats->{'test'});
	}
    }
    $res{'classified'}
    = max(abs(
#PTZ121009 this is called differently on this branch	      $classifier->apply()->get_confidences()
	      $classifier->apply()->get_values()
	      - $indata->{$prefix.'classified'})
	);
    return \%res;
}

sub _evaluate {
    my ($indata) = @_;
    my $prefix='classifier_';
    my $ctype=$indata->{$prefix.'type'};
    my $feats;
    if($indata->{$prefix.'name'} eq 'KNN'){
	$feats=&util::get_features($indata, 'distance_');
    }elsif( $ctype eq 'kernel'){
	$feats=&util::get_features($indata, 'kernel_');
    }else{
	$feats=&util::get_features($indata, $prefix);
    }
    my $machine=&_get_machine($indata, $prefix, $feats);
    #my $fun=*{'modshogun::' . $indata->{$prefix.'name'}};
    my $fun= eval('modshogun::' . $indata->{$prefix.'name'});
    if($@) {#except NameError, e:
	warn( "%s is disabled/unavailable!", $indata->{$prefix.'name'});
	return false;
    }
    # cannot refactor into function, because labels is unrefed otherwise
    my $classifier;
    if(defined($indata->{$prefix.'labels'})){
#vector_from_pdl<float64_t> is_pdl_vector(ST(0), PDL_D);
	my $labels= modshogun::BinaryLabels->new(&double($indata->{$prefix.'labels'}));
	if( $ctype eq 'kernel'){
	    $classifier=$fun->new($indata->{$prefix.'C'}, $machine, $labels);
	} elsif( $ctype eq 'linear'){
#Can't locate object method "swig_train_get" via package "modshogun::SparseRealFeatures" at /usr/src/shogun/src/interfaces/perldl_modular/modshogun.pm line 33.
	    $classifier=$fun->new($indata->{$prefix.'C'}, $feats->{'train'}, $labels);
	} elsif( $ctype eq 'knn') {
	    $classifier=$fun->new($indata->{$prefix.'k'}, $machine, $labels);
	} elsif( $ctype eq 'lda') {
	    $classifier=$fun->new($indata->{$prefix.'gamma'}, $feats->{'train'}, $labels);
	} elsif( $ctype eq 'perceptron') {
	    $classifier=$fun->new($feats->{'train'}, labels);
	} elsif( $ctype eq 'wdsvmocas'){
	    $classifier=$fun->new($indata->{$prefix.'C'}, $indata->{$prefix.'degree'},
			       $indata->{$prefix.'degree'}, $feats->{'train'}, $labels);
	} else {
	    return false;
	}
    } else {
	$classifier=$fun->new($indata->{$prefix.'C'}, $machine);
    }
    if($classifier->get_name() eq 'LibLinear'){
	print ($classifier->get_name(), " - yes\n");
	$classifier->set_liblinear_solver_type($modshogun::L2R_LR);
    }
    my $sgio =  $classifier->get_global_io();
    $sgio->enable_progress();
    $sgio->set_loglevel($modshogun::MSG_GCDEBUG);
    $sgio->set_loglevel($modshogun::MSG_INFO);

    $classifier->{parallel} = modshogun::Parallel->new();
    $classifier->{parallel}->set_num_threads($indata->{$prefix.'num_threads'});
    #PTZ121009 threads are not working..it crashes in Perl_IO_stdin...
    # but also is not translating into the ::Parrallel instance?!
    #$classifier->{parallel}->set_num_threads(1);
    if($@) {
	warn($@);
	return 0;
    }

    if($ctype eq 'linear'){
	if(defined($indata->{$prefix.'bias'})){
	    $classifier->set_bias_enabled(1);
	}
    }else{
	$classifier->set_bias_enabled(0);
    }
    if( $ctype eq 'perceptron'){
	$classifier->set_learn_rate=$indata->{$prefix.'learn_rate'};
	$classifier->set_max_iter=$indata->{$prefix.'max_iter'};
    }
    if(defined($indata->{$prefix.'epsilon'})){
	eval { $classifier->set_epsilon($indata->{$prefix.'epsilon'}); };
	if($@) {
#Can't locate object method "set_epsilon" via package "modshogun::SVMSGD" at classifier.pm line 167.
	    warn($@, Dumper($classifier));
	    #return false;
	}
    }
    if(defined($indata->{$prefix.'max_train_time'})){
	$classifier->set_max_train_time($indata->{$prefix.'max_train_time'} * 1000);
    }
    if(defined($indata->{$prefix.'linadd_enabled'})){
	$classifier->set_linadd_enabled($indata->{$prefix.'linadd_enabled'});
    }
    if(defined($indata->{$prefix.'batch_enabled'})){
	$classifier->set_batch_computation_enabled($indata->{$prefix.'batch_enabled'});
    }
    $classifier->train();

    my $res = &_get_results($indata, $prefix, $classifier, $machine, $feats);
    return &util::check_accuracy
	($res->{'accuracy'},
	 {alphas=>$res->{'alphas'}, bias=>$res->{'bias'}, sv=>$res->{'sv'},
	  classified=>$res->{'classified'}});
}

########################################################################
# public
########################################################################

sub test {
    my  ($indata) = @_;
    return &_evaluate($indata);
}

1;
__END__
