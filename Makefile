
test_all:
	python tests/test_core.py

coverage:
	coverage run tests/test_core.py
	coverage html
