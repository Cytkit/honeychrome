# Variables
PYTHON = python
BUILD_SCRIPT = build.py
INSTALL_SCRIPT = install_linux.sh

# Detect if we are inside a venv locally.
# If .venv exists, use that path; otherwise use system python (perfect for GitHub)
ifeq ($(shell test -f .venv/bin/python && echo yes),yes)
	PYTHON = .venv/bin/python
endif

.PHONY: help venv build install clean

help:
	@echo "HoneyChrome Build System"
	@echo "  make venv    - Create virtual environment and install deps"
	@echo "  make build   - Run the PyInstaller build script"
	@echo "  make install - Register the app in the Linux app menu"
	@echo "  make clean   - Remove build artifacts and logs"

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) install_annoy.py
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pyinstaller

build:
	@echo "Starting PyInstaller Build..."
	$(PYTHON) $(BUILD_SCRIPT)

install:
	@echo "Installing Desktop Entry..."
	@bash $(INSTALL_SCRIPT)

clean:
	rm -rf build/ dist/ __pycache__/ *.spec
	@echo "Cleaned up build artifacts."

# Variables for packaging
VERSION = 0.8.3
PKG_NAME = honeychrome-v$(VERSION)-linux-x64

package: build
	@echo "Creating portable package..."
	mkdir -p $(PKG_NAME)
	# Copy the executable
	cp -r dist/honeychrome $(PKG_NAME)/
	# Copy the install script and icon so they can install it locally
	cp install_linux.sh $(PKG_NAME)/
	# Copy README
	cp README.md $(PKG_NAME)/
	# Compress
	tar -czvf $(PKG_NAME).tar.gz $(PKG_NAME)
	# Cleanup folder
	rm -rf $(PKG_NAME)
	@echo "Package created: $(PKG_NAME).tar.gz"