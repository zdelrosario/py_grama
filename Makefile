
test:
	cd tests; python -m unittest discover

coverage:
	cd tests; coverage run -m unittest discover
	cd tests; coverage html
	xdg-open htmlcov/index.html
