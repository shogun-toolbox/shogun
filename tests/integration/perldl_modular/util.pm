package util;

use modshogun;

=pod

use Shogun.Features;use Shogun.Preprocessor;
use Shogun.Distance;use Shogun.Kernel;

=cut

use PDL;
#use PDL::NiceSlice; 

use Data::Dumper;

use base qw(Exporter);
our %EXPORT_TAGS =
(
 #FIELDS => [ @EXPORT_OK, @EXPORT ]
 feats => [qw(check_accuracy get_args get_features get_feats_simple get_feats_string get_feats_string_complex )]
 , basic => [qw()]
 , misc => [qw(add_preprocessor)]
);

Exporter::export_tags(qw//);
Exporter::export_ok_tags(qw/feats basic misc/);

sub check_accuracy {
    my ($accuracy, $kwargs) = @_;
    my $acc=&double($accuracy);
    my @output;
    while( my ($key, $val) = each(%$kwargs)) {
	if($val) {
	    push(@output, sprintf('%s: %e', $key, $val));
	}
    }
    printf(join(', ', @output) . ' <--- accuracy: %e', $accuracy);
    foreach my $val (values(%$kwargs)) {
	if($val > $acc) {
	    return 0;#false
	}
    }
    return 1;#true
}

sub get_args {
    my ($indata, $prefix) = @_;

=head3 get_args

	Slightly esoteric function to build a tuple to be used as argument to
	constructor calls.

	Perl dicts are not ordered, so we have to look at the number in
	the parameter's name and insert items appropriately into an
	ordered list

=cut
    
    my $ident=$prefix . 'arg';
    # need to pregenerate list for using indices in loop
    my @args = ();
    foreach my $i (keys(%$indata)) {
	unless($i =~ /$ident/) { next;}
	my ($idx) = $i =~ m/$ident(\d+)/;
	unless($idx >= 0) {
	    warn( 'Wrong indata data %s: "%s"!' , $ident, $i);
	}
	if($i =~ /_distance/) { # DistanceKernel
	    $args[$idx] = eval('modshogun::' . $indata->{$i})->new();
	}else{
	    $args[$idx] = eval($indata->{$i});
	    #except TypeError: # no bool
	    if($@) {
		$args[$idx] = $indata->{$i};
	    }
	    #False  and True...
	    if($args[$idx] eq 'False') {
		$args[$idx] = 0;#$modshogun::False;
	    } elsif($args[$idx] eq 'True') {
		$args[$idx] = 1;#$modshogun::True;
	    }
	}
    }
    # weed out superfluous Nones
    @args = grep(defined($_), @args);
    return \@args; #&filter(lambda arg: arg is not None, $args);
}

sub get_features
{
    my ($indata, $prefix) = @_;
    my $fclass=$prefix.'feature_class';
    if($indata->{$fclass} eq 'simple') {
	return &get_feats_simple($indata, $prefix);
    } elsif($indata->{$fclass} eq 'string') {
	return &get_feats_string($indata, $prefix);
    }elsif ($indata->{$fclass} eq 'string_complex') {
	return &get_feats_string_complex($indata, $prefix);
    } elsif($indata->{$fclass} eq 'wd') {
	return &get_feats_wd($indata, $prefix);
    } else {
	croak('Unknown feature class %s!', $indata->{$prefix.'feature_class'});
    }
}

sub get_feats_simple {
    my ($indata, $prefix) = @_;

    my $ftype=$indata->{$prefix.'feature_type'};

    # have to explicitely set data type for numpy if not real
    my %as_types=(
	'Byte'=> \&byte,
	'Real'=> \&double,
	'Word'=> \&ushort
	, 'Char' => \&byte
	);
    unless(defined($as_types{$ftype})) {
	croak("PDL type conversion $ftype not found");
    }
    my $data_train = $as_types{$ftype}->($indata->{$prefix.'data_train'});
    my $data_test = $as_types{$ftype}->($indata->{$prefix.'data_test'});
    my $ftrain;
    my $ftest;
    if($ftype eq 'Byte' or $ftype eq 'Char') {
	my $alphabet= ${'modshogun::' . $indata->{$prefix.'alphabet'}};
	$ftrain=eval('modshogun::' . $ftype . 'Features')->new($alphabet);
	$ftest =eval('modshogun::' . $ftype . 'Features')->new($alphabet);
	$ftrain->copy_feature_matrix($data_train);
	$ftest->copy_feature_matrix($data_test);
    } else {
#tests (for 'Real' type) matrix_from_pdl<float64_t>(arg1, ST(0), PDL_D)) ; is_pdl_matrix(ST(0), PDL_D);
	$ftrain = eval('modshogun::' . $ftype . 'Features')->new($data_train);
	$ftest  = eval('modshogun::' . $ftype . 'Features')->new($data_test);
    }
    if($indata->{$prefix.'name'} =~ /Sparse/ or (
	   defined($indata->{'classifier_type'}) and 
	   $indata->{'classifier_type'} eq 'linear')
	) {
#  _v = is_pdl_sparse_matrix(ST(0), PDL_D);
	my $sparse_train=eval('modshogun::' .'Sparse'.$ftype.'Features')->new();
	$sparse_train->obtain_from_simple($ftrain);

	my $sparse_test=eval('modshogun::' .'Sparse'.$ftype.'Features')->new();
	$sparse_test->obtain_from_simple($ftest);

	return {'train' => $sparse_train, 'test' => $sparse_test};
    }else{
	return {'train' => $ftrain, 'test' => $ftest};
    }
}

sub get_feats_string {
    my ($indata, $prefix) = @_;
    my $ftype=$indata->{$prefix.'feature_type'};
    my $alphabet=${'modshogun::' . $indata->{$prefix.'alphabet'}};
    my %feats=(
	'train'=> eval('modshogun::' . 'String'.$ftype.'Features')->new($alphabet)
	,'test'=> eval('modshogun::' . 'String'.$ftype.'Features')->new($alphabet)
	);
    #$feats{'train'}->set_features($indata->{$prefix.'data_train'}->slice('0:-1,(0)'));
    #$feats{'test'}->set_features($indata->{$prefix.'data_test'}->slice('0:-1,(0)'));
    $feats{'train'}->set_features($indata->{$prefix.'data_train'});
    $feats{'test'}->set_features($indata->{$prefix.'data_test'});
    return \%feats;
}
sub get_feats_string_complex
{
    my ($indata, $prefix) = @_;
    my $alphabet = ${'modshogun::' . $indata->{$prefix.'alphabet'}};
    my %feats=(
	'train'=> modshogun::StringCharFeatures->new($alphabet)
	, 'test'=> modshogun::StringCharFeatures->new($alphabet)
	);
#PTZ121011 ->server might be needed to be sure it is a dense object.
    #my $data_train = $indata->{$prefix.'data_train'}->slice('0:-1,(0)'); #->nslice([0,-1], 0); 
    #my $data_test  = $indata->{$prefix.'data_test'}->slice('0:-1,(0)'); #->nslice([0,-1], 0);
    my $data_train = $indata->{$prefix.'data_train'};
    my $data_test  = $indata->{$prefix.'data_test'};
    if($alphabet == $modshogun::CUBE) # data_{train,test} ints due to test.py:_read_matrix
    { ##map { $a($_ - 1) .= $_; } (1..$a->nelem);    # Lots of little ops
	#map { $data_train->nslice($_) = chr($data_train->nslice($_)) } (0..$data_train->nelem - 1);
	#map { $data_test->nslice($_) = chr($data_test->nslice($_)) } (0..$data_test->nelem - 1);
	
    }
    $feats{'train'}->set_features($data_train);
    $feats{'test'}->set_features($data_test);

    my $feat=eval('modshogun::' .'String'.$indata->{$prefix.'feature_type'}."Features")->new($alphabet);
    $feat->obtain_from_char($feats{'train'}, $indata->{$prefix.'order'}-1,
			    $indata->{$prefix.'order'}, $indata->{$prefix.'gap'},
			    ${'modshogun::' . $indata->{$prefix.'reverse'}});
    $feats{'train'}=$feat;

    $feat=eval('modshogun::' .'String'.$indata->{$prefix.'feature_type'}."Features")->new($alphabet);
    $feat->obtain_from_char($feats{'test'}, $indata->{$prefix.'order'}-1,
			    $indata->{$prefix.'order'},  $indata->{$prefix.'gap'},
			    ${'modshogun::' . $indata->{$prefix.'reverse'}});
    $feats{'test'}=$feat;

    if( $indata->{$prefix.'feature_type'} eq 'Word' or 
	$indata->{$prefix.'feature_type'} eq 'Ulong'){
	my $name = 'Sort' .$indata->{$prefix.'feature_type'}.'String';
	return &add_preprocessor($name, \%feats);
    } else {
	return \%feats;
    }
}

sub get_feats_wd
{
    my ($indata, $prefix) = @_;
    my $order=$indata->{$prefix.'order'};
    my %feats;

    my $charfeat=&modshogun::StringCharFeatures(&modshogun::DNA);
    my $charfeat->set_features(@{$indata->{$prefix.'data_train'}[0]});
    my $bytefeat=&modshogun::StringByteFeatures(&modshogun::RAWDNA);
    $bytefeat->obtain_from_char($charfeat, 0, 1, 0, false);
    $feats{'train'}=&modshogun::WDFeatures($bytefeat, $order, $order);

    $charfeat=&modshogun::StringCharFeatures(&modshogun::DNA);
    $charfeat->set_features(@{$indata->{$prefix.'data_test'}[0]});
    $bytefeat=&modshogun::StringByteFeatures(&modshogun::RAWDNA);
    $bytefeat->obtain_from_char($charfeat, 0, 1, 0, false);
    $feats{'test'}=&modshogun::WDFeatures($bytefeat, $order, $order);

    return \%feats;
}

sub add_preprocessor
{
    my ($name, $feats, $args) = @_;
    #my $fun=*{$name};
    #my $preproc=*{$name.'::new'}->($name, @$args);
    my $preproc=eval('modshogun::' . $name)->new(@$args);
    $preproc->init($feats->{'train'});
    $feats->{'train'}->add_preprocessor($preproc);
    $feats->{'train'}->apply_preprocessor();
    $feats->{'test'}->add_preprocessor($preproc);
    $feats->{'test'}->apply_preprocessor();
    return $feats;
}

1;
__END__
=head1

Utilities for testing

=cut

from shogun.Features import *
from shogun.Preprocessor import *
from shogun.Distance import *
from shogun.Kernel import *
from numpy import *


def check_accuracy (accuracy, **kwargs):
	acc=double(accuracy)
	output=[]

	for key, val in kwargs.iteritems():
		if val is not None:
			output.append('%s: %e' % (key, val))
	print ', '.join(output)+' <--- accuracy: %e' % accuracy

	for val in kwargs.itervalues():
		if val>acc:
			return False

	return True


def get_args (indata, prefix=''):
	"""
	Slightly esoteric function to build a tuple to be used as argument to
	constructor calls.

	Python dicts are not ordered, so we have to look at the number in
	Perl dicts are not ordered, so we have to look at the number in
	the parameter's name and insert items appropriately into an
	ordered list
	"""

	ident=prefix+'arg'
	# need to pregenerate list for using indices in loop
	args=len(indata)*[None]

	for i in indata:
		if i.find(ident)==-1:
			continue

		try:
			idx=int(i[len(ident)])
		except ValueError:
			raise ValueError, 'Wrong indata data %s: "%s"!' % (ident, i)

		if i.find('_distance')!=-1: # DistanceKernel
			args[idx]=eval(indata[i]+'()')
		else:
			try:
				args[idx]=eval(indata[i])
			except TypeError: # no bool
				args[idx]=indata[i]

	# weed out superfluous Nones
	return filter(lambda arg: arg is not None, args)


def get_features(indata, prefix=''):
	fclass=prefix+'feature_class'
	if indata[fclass]=='simple':
		return get_feats_simple(indata, prefix)
	elif indata[fclass]=='string':
		return get_feats_string(indata, prefix)
	elif indata[fclass]=='string_complex':
		return get_feats_string_complex(indata, prefix)
	elif indata[fclass]=='wd':
		return get_feats_wd(indata, prefix)
	else:
		raise ValueError, \
			'Unknown feature class %s!'%$indata->{$prefix.'feature_class']


def get_feats_simple (indata, prefix=''):
	ftype=$indata->{$prefix.'feature_type']

	# have to explicitely set data type for numpy if not real
	as_types={
		'Byte': ubyte,
		'Real': double,
		'Word': ushort
	}
	data_train=$indata->{$prefix.'data_train'].astype(as_types[ftype])
	data_test=$indata->{$prefix.'data_test'].astype(as_types[ftype])

	if ftype=='Byte' or ftype=='Char':
		alphabet=eval($indata->{$prefix.'alphabet'])
		ftrain=eval(ftype+'Features(alphabet)')
		ftest=eval(ftype+'Features(alphabet)')
		ftrain.copy_feature_matrix(data_train)
		ftest.copy_feature_matrix(data_test)
	else:
		ftrain=eval(ftype+'Features(data_train)')
		ftest=eval(ftype+'Features(data_test)')

	if ($indata->{$prefix.'name'].find('Sparse')!=-1 or (
		indata.has_key('classifier_type') and \
			indata['classifier_type']=='linear')):
		sparse_train=eval('Sparse'+ftype+'Features()')
		sparse_train.obtain_from_simple(ftrain)

		sparse_test=eval('Sparse'+ftype+'Features()')
		sparse_test.obtain_from_simple(ftest)

		return {'train':sparse_train, 'test':sparse_test}
	else:
		return {'train':ftrain, 'test':ftest}


def get_feats_string (indata, prefix=''):
	ftype=$indata->{$prefix.'feature_type']
	alphabet=eval($indata->{$prefix.'alphabet'])
	feats={
		'train': eval('String'+ftype+'Features(alphabet)'),
		'test': eval('String'+ftype+'Features(alphabet)')
	}
	feats['train'].set_features(list(indata[prefix+'data_train'][0]))
	feats['test'].set_features(list(indata[prefix+'data_test'][0]))

	return feats


def get_feats_string_complex (indata, prefix=''):
	alphabet=eval(indata[prefix+'alphabet'])
	feats={
		'train': StringCharFeatures(alphabet),
		'test': StringCharFeatures(alphabet)
	}

	if alphabet==CUBE: # data_{train,test} ints due to test.py:_read_matrix
		data_train=[str(x) for x in list(indata[prefix+'data_train'][0])]
		data_test=[str(x) for x in list(indata[prefix+'data_test'][0])]
	else:
		data_train=list(indata[prefix+'data_train'][0])
		data_test=list(indata[prefix+'data_test'][0])

	feats['train'].set_features(data_train)
	feats['test'].set_features(data_test)

	feat=eval('String'+indata[prefix+'feature_type']+ \
		"Features(alphabet)")
	feat.obtain_from_char(feats['train'], indata[prefix+'order']-1,
		indata[prefix+'order'], indata[prefix+'gap'],
		eval(indata[prefix+'reverse']))
	feats['train']=feat

	feat=eval('String'+indata[prefix+'feature_type']+ \
		"Features(alphabet)")
	feat.obtain_from_char(feats['test'], indata[prefix+'order']-1,
		indata[prefix+'order'], indata[prefix+'gap'],
		eval(indata[prefix+'reverse']))
	feats['test']=feat

	if indata[prefix+'feature_type']=='Word' or \
		indata[prefix+'feature_type']=='Ulong':
		name='Sort'+indata[prefix+'feature_type']+'String'
		return add_preprocessor(name, feats)
	else:
		return feats


def get_feats_wd (indata, prefix=''):
	order=indata[prefix+'order']
	feats={}

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(list(indata[prefix+'data_train'][0]))
	bytefeat=StringByteFeatures(RAWDNA)
	bytefeat.obtain_from_char(charfeat, 0, 1, 0, False)
	feats['train']=WDFeatures(bytefeat, order, order)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(list(indata[prefix+'data_test'][0]))
	bytefeat=StringByteFeatures(RAWDNA)
	bytefeat.obtain_from_char(charfeat, 0, 1, 0, False)
	feats['test']=WDFeatures(bytefeat, order, order)

	return feats


def add_preprocessor(name, feats, *args):
	fun=eval(name)
	preproc=fun(*args)
	preproc.init(feats['train'])
	feats['train'].add_preprocessor(preproc)
	feats['train'].apply_preprocessor()
	feats['test'].add_preprocessor(preproc)
	feats['test'].apply_preprocessor()

	return feats

