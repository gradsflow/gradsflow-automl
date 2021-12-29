build-docs:
	cp README.md docs/index.md

docsserve:
	mkdocs serve --dirtyreload --livereload

test:
	python tests/__init__.py
	pytest

coverage:  ## Run tests with coverage
		coverage erase
		coverage run -m pytest
		coverage report -m
		coverage xml

clean:
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

style:
	black .
	isort --profile black .

build: clean
	python -m build

test_pypi: build
	twine upload -r testpypi dist/*

pypi: build
	twine upload dist/*

push:
	git push && git push --tags

install: style clean
	flit install --deps none
