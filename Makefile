
test:
	cd tests; python -m unittest discover

coverage:
	cd tests; coverage run -m unittest discover
	coverage html
	xdg-open htmlcov/index.html
