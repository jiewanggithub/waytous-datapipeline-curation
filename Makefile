.PHONY: install test demo docker-build

install:
	pip install -e '.[dev]'

test:
	pytest

demo:
	waytous-curate --input examples/samples.jsonl --output output/demo --dataset-version demo-v1

docker-build:
	docker build -t waytous-pointcloud-curation:latest .

