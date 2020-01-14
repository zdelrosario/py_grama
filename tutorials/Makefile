SRC := $(wildcard *master.ipynb)
TARS := $(subst master,assignment,$(SRC))

all: ${TARS}

%assignment.ipynb: %master.ipynb
	./sep.py $<

strip:
	nbstripout *solution.ipynb
	nbstripout *assignment.ipynb

.PHONY: clean
clean:
	rm -f *solution.ipynb
	rm -f *assignment.ipynb
