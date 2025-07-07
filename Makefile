.PHONY: install test

# Install your dependencies
install:
	pip install -r Requirements.txt

# Run your pytest suite
test:
	pytest Code/tests --maxfail=1 --disable-warnings -q