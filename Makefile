
test:
	cd tests; python -m unittest discover

coverage:
	cd tests; coverage run -m unittest discover
	cd tests; coverage html
	open tests/htmlcov/index.html

install:
	python setup.py install
