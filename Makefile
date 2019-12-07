
test_all:
	python tests/test_pipes.py
	python tests/test_core.py
	python tests/test_evals.py

coverage:
	cd tests; coverage run -m unittest discover
	coverage html
	xdg-open htmlcov/index.html
