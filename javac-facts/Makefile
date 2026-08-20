# zemble-javac-facts: a standalone javac plugin, built with plain javac/jar.

JAVAC ?= javac
JAR ?= jar
PYTHON ?= python3
RELEASE ?= 17

SRC_DIR := src/main/java
RES_DIR := src/main/resources
BUILD_DIR := build
CLASSES_DIR := $(BUILD_DIR)/classes
JAR_FILE := $(BUILD_DIR)/zemble-javac-facts.jar

TEST_DIR := test
FIXTURE_DIR := $(TEST_DIR)/fixtures
TEST_BUILD := $(BUILD_DIR)/test
FACTS_FILE := $(TEST_BUILD)/facts.jsonl

SOURCES := $(shell find $(SRC_DIR) -name '*.java')
FIXTURES := $(shell find $(FIXTURE_DIR) -name '*.java')

.PHONY: build test clean

build: $(JAR_FILE)

$(JAR_FILE): $(SOURCES) $(RES_DIR)/META-INF/services/com.sun.source.util.Plugin
	@mkdir -p $(CLASSES_DIR)
	$(JAVAC) --release $(RELEASE) -Xlint:all -d $(CLASSES_DIR) $(SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(JAR) --create --file $(JAR_FILE) -C $(CLASSES_DIR) . -C $(RES_DIR) META-INF

test: build
	@rm -rf $(TEST_BUILD)
	@mkdir -p $(TEST_BUILD)/classes
	$(JAVAC) -proc:none -processorpath $(JAR_FILE) \
		-Xplugin:"ZembleFacts out=$(FACTS_FILE) root=$(FIXTURE_DIR)" \
		-d $(TEST_BUILD)/classes $(FIXTURES)
	$(PYTHON) $(TEST_DIR)/check_facts.py $(FACTS_FILE)

clean:
	rm -rf $(BUILD_DIR)
