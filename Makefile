PYTHON_BASE := /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
PYTHON := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,$(PYTHON_BASE))
QT_CMAKE := /opt/Qt/6.11.2/macos/lib/cmake/Qt6
BUILD_DIR := build

.PHONY: setup configure build gui test smoke clean

setup:
	$(PYTHON_BASE) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install --no-deps -e .

configure:
	cmake -S . -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_PREFIX_PATH=$(QT_CMAKE) \
		-DIPDE_PYTHON_EXECUTABLE=$(PYTHON)

build: configure
	cmake --build $(BUILD_DIR)

gui: build
	open "$(CURDIR)/$(BUILD_DIR)/IPDE.app"

test:
	PYTHONPATH=$(CURDIR)/src $(PYTHON) -m unittest discover -s tests -v

smoke: build
	"$(CURDIR)/$(BUILD_DIR)/IPDE.app/Contents/MacOS/IPDE" --smoke-test

clean:
	cmake -E remove_directory $(BUILD_DIR)
