LATEXMK ?= latexmk
LATEXMKRC ?= latexmkrc

.PHONY: pdf anonymous clean

pdf:
	$(LATEXMK) -r $(LATEXMKRC) -pdf main.tex

anonymous:
	$(LATEXMK) -r $(LATEXMKRC) -pdf main_anonymous.tex

clean:
	$(LATEXMK) -r $(LATEXMKRC) -C main.tex
	$(LATEXMK) -r $(LATEXMKRC) -C main_anonymous.tex
	-if exist build rmdir /s /q build
