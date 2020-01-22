
test:
	cd tests; python -m unittest discover

coverage:
	cd tests; coverage run -m unittest discover
	cd tests; coverage html
	open tests/htmlcov/index.html

dist:
	python setup.py sdist bdist_wheel

upload:
	twine upload --repository pypi dist/*.tar.gz

install:
	python setup.py install

.PHONY: dist
