
test:
	cd tests; python -m unittest discover

test-all:
	cd tests; python -m unittest discover
	cd tutorials; python -m unittest discover

coverage:
	cd tests; coverage run -m unittest discover
	cd tests; coverage html
	open tests/htmlcov/index.html

doc:
	sphinx-build -b html docs build

autodoc:
	rm docs/source/grama*
	rm docs/source/modules.rst
	cd docs; sphinx-apidoc -o source/ ../grama

dist:
	python setup.py sdist bdist_wheel

upload:
	python -m twine upload --repository pypi dist/*.tar.gz

install:
	python setup.py install

.PHONY: dist
