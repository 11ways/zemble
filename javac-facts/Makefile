# zemble-javac-facts: a standalone javac plugin, built with plain javac/jar.

JAVAC ?= javac
JAVA ?= java
JAR ?= jar
PYTHON ?= python3
RELEASE ?= 17

SRC_DIR := src/main/java
RES_DIR := src/main/resources
MANIFEST := src/main/manifest/MANIFEST.MF
BUILD_DIR := build
CLASSES_DIR := $(BUILD_DIR)/classes
JAR_FILE := $(BUILD_DIR)/zemble-javac-facts.jar

TEST_DIR := test
FIXTURE_DIR := $(TEST_DIR)/fixtures
TEST_BUILD := $(BUILD_DIR)/test
FACTS_FILE := $(TEST_BUILD)/facts.jsonl
CLASSES_FACTS_FILE := $(TEST_BUILD)/facts-classes.jsonl
JAVAC_EXPORTS := -J--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED \
	-J--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED

SOURCES := $(shell find $(SRC_DIR) -name '*.java')
FIXTURES := $(shell find $(FIXTURE_DIR) -name '*.java')

.PHONY: build test clean

build: $(JAR_FILE)

$(JAR_FILE): $(SOURCES) $(RES_DIR)/META-INF/services/com.sun.source.util.Plugin $(MANIFEST)
	@mkdir -p $(CLASSES_DIR)
	$(JAVAC) --release $(RELEASE) -Xlint:all -d $(CLASSES_DIR) $(SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(JAR) --create --file $(JAR_FILE) --manifest $(MANIFEST) -C $(CLASSES_DIR) . -C $(RES_DIR) META-INF

test: build
	@rm -rf $(TEST_BUILD)
	@mkdir -p $(TEST_BUILD)/classes
	$(JAVAC) -proc:none -processorpath $(JAR_FILE) \
		-Xplugin:"ZembleFacts out=$(FACTS_FILE) root=$(FIXTURE_DIR)" \
		-d $(TEST_BUILD)/classes $(FIXTURES)
	$(PYTHON) $(TEST_DIR)/check_facts.py $(FACTS_FILE)
	@# base=classes: out and root relative to -d, the one constant argument a build tool passes.
	@mkdir -p $(TEST_BUILD)/classes-base
	$(JAVAC) $(JAVAC_EXPORTS) -proc:none -processorpath $(JAR_FILE) \
		-Xplugin:"ZembleFacts base=classes out=../facts-classes.jsonl root=../../../$(FIXTURE_DIR)" \
		-d $(TEST_BUILD)/classes-base $(FIXTURES)
	$(PYTHON) $(TEST_DIR)/check_facts.py $(CLASSES_FACTS_FILE)
	@# Without javac's exported context base=classes writes nothing and never fails the compile.
	@mkdir -p $(TEST_BUILD)/classes-unexported
	$(JAVAC) -proc:none -processorpath $(JAR_FILE) \
		-Xplugin:"ZembleFacts base=classes out=../facts-unexported.jsonl root=../../../$(FIXTURE_DIR)" \
		-d $(TEST_BUILD)/classes-unexported $(FIXTURES)
	test ! -e $(TEST_BUILD)/facts-unexported.jsonl
	$(JAVA) --class-path $(JAR_FILE) $(TEST_DIR)/MergeFacts.java $(FIXTURE_DIR) $(TEST_BUILD)/merge

clean:
	rm -rf $(BUILD_DIR)
