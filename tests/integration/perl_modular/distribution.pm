package distribution;
use util;
use PDL;
use modshogun;

sub _evaluate
{
    my  ($indata) = @_;
    my $prefix = 'distribution_';
    my $feats = &util::get_features($indata, $prefix);
    my $distribution;
    if($indata->{$prefix.'name'} eq 'HMM'){
	$distribution = modshogun::HMM->new($feats->{'train'}, $indata->{$prefix.'N'},
					    $indata->{$prefix.'M'}, $indata->{$prefix.'pseudo'});
	$distribution->train();
	$distribution->baum_welch_viterbi_train($modshogun::BW_NORMAL);
    }else{
	my $dfun = eval('modshogun::' . $indata->{$prefix.'name'});
	$distribution = $dfun->new($feats->{'train'});
	$distribution->train();
    }
    my $likelihood = $distribution->get_log_likelihood_sample();
    my $num_examples = $feats->{'train'}->get_num_vectors();
    my $num_param = $distribution->get_num_model_parameters();
    my $derivatives = 0;
    foreach my $i (0..$num_param-1) {
	foreach my $j (0..$num_examples-1) {
	    my $val = $distribution->get_log_derivative($i, $j);
#PTZ121013 SG or PDL representation of NAN and INF PDL::Type::badvalue, PDL::Type::usenan
	    if( $val ne 'inf' and $val ne 'nan') { # only consider sparse matrix!
		$derivatives += $val;
	    }
	}
    }
    $derivatives = abs($derivatives-$indata->{$prefix.'derivatives'});
    $likelihood = abs($likelihood-$indata->{$prefix.'likelihood'});

    if( $indata->{$prefix.'name'} eq 'HMM'){
	my $best_path = 0;
	my $best_path_state = 0;
	foreach my $i (0..$indata->{$prefix.'num_examples'}-1) {
	    $best_path+=$distribution->best_path($i);
	    foreach my $j (0..$indata->{$prefix.'N'}-1) {
		$best_path_state += $distribution->get_best_path_state($i, $j);
	    }
	}
	$best_path=abs($best_path - $indata->{$prefix.'best_path'});
	$best_path_state=abs($best_path_state -
			     $indata->{$prefix.'best_path_state'});

	return &util::check_accuracy($indata->{$prefix.'accuracy'}
				     , {
					 derivatives=>$derivatives
					     , likelihood=>$likelihood
					     , best_path=>$best_path
					     , best_path_state=>$best_path_state
				     }
	    );
    } else {
	return &util::check_accuracy($indata->{$prefix.'accuracy'}, {
	    derivatives=>$derivatives, likelihood=>$likelihood});
    }
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

=head2

 Test Distribution

=cut
