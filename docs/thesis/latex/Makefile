# ===========================================================================
#  Makefile per la tesi
#
#   make          compila main.pdf (latexmk gestisce le passate e biber)
#   make watch    ricompila a ogni salvataggio
#   make clean    rimuove gli ausiliari, tiene il PDF
#   make distclean rimuove anche il PDF
#   make final    compila disattivando le annotazioni todonotes
#   make check    elenca le annotazioni ancora aperte nei sorgenti
#   make todo     conteggio delle annotazioni per tipo e per file
#   make wc       conteggio approssimativo delle parole per capitolo
# ===========================================================================

MAIN    := main
LATEXMK := latexmk -pdf -interaction=nonstopmode -file-line-error

.PHONY: all watch clean distclean final wc check todo

all:
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	latexmk -c
	rm -f *.bbl *.run.xml *.synctex.gz capitoli/*.aux appendici/*.aux parti/*.aux

distclean:
	latexmk -C
	rm -f *.bbl *.run.xml *.synctex.gz capitoli/*.aux appendici/*.aux parti/*.aux

# Compilazione per la consegna: disattiva le annotazioni di lavorazione.
final:
	@sed -i.bak 's/^\\usepackage\[textwidth/\\usepackage[disable, textwidth/' preambolo.tex
	$(LATEXMK) $(MAIN).tex || true
	@mv preambolo.tex.bak preambolo.tex
	@echo "PDF finale prodotto; preambolo.tex ripristinato."

# Conteggio parole (richiede texcount, incluso in texlive-extra-utils).
wc:
	@texcount -inc -total -sum $(MAIN).tex 2>/dev/null || \
	  echo "texcount non trovato: sudo apt install texlive-extra-utils"

# Elenca le annotazioni ancora aperte nei SORGENTI (esclude .aux e altri
# ausiliari, che contengono copie del testo e falsavano il conteggio).
SOURCES := $(wildcard capitoli/*.tex appendici/*.tex parti/*.tex) metadati.tex

check:
	@grep -n '\\SCR{\|\\DA{\|\\VER{\|<<' $(SOURCES) \
	  | sed 's/:[[:space:]]*/: /' || echo "nessuna annotazione aperta"

# Conteggio delle annotazioni per tipo e per file.
todo:
	@printf "%-6s %s\n" SCR "$$(grep -oh '\\SCR{' $(SOURCES) | wc -l)"
	@printf "%-6s %s\n" DA  "$$(grep -oh '\\DA{'  $(SOURCES) | wc -l)"
	@printf "%-6s %s\n" VER "$$(grep -oh '\\VER{' $(SOURCES) | wc -l)"
	@echo "---"
	@for f in $(SOURCES); do \
	  n=$$(grep -c '\\SCR{\|\\DA{\|\\VER{' $$f); \
	  [ "$$n" -gt 0 ] && printf "%-42s %3s\n" "$$f" "$$n"; \
	done; true
