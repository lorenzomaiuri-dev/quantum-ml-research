# Configurazione latexmk per la tesi.
$pdf_mode = 1;                 # pdflatex
$bibtex_use = 2;               # esegui biber e ripulisci i .bbl
$biber = 'biber %O %S';
$out_dir = '.';
$clean_ext = 'bbl run.xml synctex.gz nav snm vrb tdo';
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
