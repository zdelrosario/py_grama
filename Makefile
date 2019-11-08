
test_all:
	python tests/test_core.py
	python tests/test_evals.py

coverage:
	coverage run tests/test_core.py
	coverage run tests/test_evals.py
	coverage html
	xdg-open htmlcov/index.html
