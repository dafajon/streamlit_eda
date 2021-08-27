APP_NAME=steda
.PHONY: build

build: 
	@echo 'Building $(APP_NAME):$(VERSION)'
	docker build -t $(APP_NAME) -f Dockerfile .
