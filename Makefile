.PHONY: build test sanitize tsan python-test bench format docker

build:
	cmake --preset release && cmake --build --preset release

test: build
	ctest --preset release

sanitize:
	cmake --preset sanitizer && cmake --build --preset sanitizer && ctest --preset sanitizer

tsan:
	cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan

python-test:
	pip install . && cd /tmp && python3 -m pytest -q $(CURDIR)/tests/python

# Reproduces every benchmark number in docs/RESULTS.md (takes about an hour, downloads ~1 GB).
bench:
	bash bench/run_all.sh

format:
	git ls-files '*.cpp' '*.hpp' | xargs clang-format -i

docker:
	docker compose up --build
