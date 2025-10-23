use strict;
use warnings;
use Cwd;
use File::Spec;

$out_dir = 'build';
$aux_dir = 'build';
$pdf_mode = 1;
$dvi_mode = $postscript_mode = 0;

unless (-d $out_dir) {
    mkdir $out_dir or warn "Couldn't create directory '$out_dir': $!";
}

my $abs_out_dir = File::Spec->rel2abs($out_dir, Cwd::getcwd());
$ENV{'TEXMF_OUTPUT_DIRECTORY'} = $abs_out_dir;

my $pdflatex_version = '';
{
    my $out = `pdflatex --version 2>&1`;
    $pdflatex_version = $out // '';
}

if ($pdflatex_version =~ /MiKTeX/i) {
    $pdflatex = 'pdflatex --enable-write18 -interaction=nonstopmode -halt-on-error %O %S';
} else {
    $pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -halt-on-error %O %S';
}

1;
