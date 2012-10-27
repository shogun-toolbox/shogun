package clustering;
use util;
use PDL;
use modshogun;

sub _evaluate
{
    my ($indata) = @_;
    my $first_arg;
    if(defined( $indata->{'clustering_k'})) {
	$first_arg = $indata->{'clustering_k'};
    }elsif(defined($indata->{'clustering_merges'})) {
	$first_arg = $indata->{'clustering_merges'};
    }else{
	return false;
    }
    my $feats = &util::get_features($indata, 'distance_');
    my $dfun = eval('modshogun::' . $indata->{'distance_name'});
    my $distance = $dfun->new($feats->{'train'}, $feats->{'train'});

    my $cfun = eval('modshogun::' . $indata->{'clustering_name'});
    my $clustering = $cfun->new($first_arg, $distance);
    $clustering->train();

    if(defined($indata->{'clustering_radi'})) {
	my $radi=max(abs($clustering->get_radiuses() - $indata->{'clustering_radi'}));
	my $centers=max(abs($clustering->get_cluster_centers()
			    - $indata->{'clustering_centers'})->flat());
	return(&util::check_accuracy(
		    $indata->{'clustering_accuracy'}
		    , {radi=>$radi, centers=>$centers}));
    } elsif(defined($indata->{'clustering_merge_distance'})) {
	my $merge_distance = max(abs($clustering->get_merge_distances()
				   - $indata->{'clustering_merge_distance'}));
	$pairs = max(abs($clustering->get_cluster_pairs()
		       - $indata->{'clustering_pairs'})->flat());
	return &util::check_accuracy($indata->{'clustering_accuracy'}
				     , {merge_distance=>$merge_distance, pairs=>$pairs}
	    );
    } else {
	return &util::check_accuracy($indata->{'clustering_accuracy'});
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

