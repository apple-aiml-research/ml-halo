#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# halo_comm codegen + version-pin notes.
#
# The generated *_pb2.py carry a gencode version stamp; at import time
# google.protobuf enforces RUNTIME >= GENCODE (same major). This project pins
# the runtime to protobuf 5.29.x — halo_comm python/pyproject.toml declares
# "protobuf>=5.29,<6". So generate with protoc RELEASE 29.x, which stamps gencode 5.29.x;
# that loads under any in-spec 5.29 runtime.
#
# Do NOT use a newer protoc: brew's protoc 35 stamps gencode 7.x, which a 5.29
# runtime refuses to load ("runtime version cannot be older than the linked
# gencode version"). protoc has no Homebrew-pinnable 29 formula.
#
# Easiest: run `make protoc29` once. It downloads the pinned protoc 29 for your
# OS/arch into .tooling/ (git-ignored, no PATH changes), and `make protos` then
# uses it automatically — and hard-fails on any other protoc.

PY_OUT    := python/halo_comm/msgs
PROTOS    := $(wildcard msgs/*.proto)
PROTOC    ?= protoc

# Pinned protoc bootstrapped by `make protoc29` lands here; `make protos` uses it
# automatically if present (overridable with PROTOC=...).
PROTOC_VERSION := 29.0
PROTOC_DIR     := $(CURDIR)/.tooling/protoc-$(PROTOC_VERSION)
PROTOC_BIN     := $(PROTOC_DIR)/bin/protoc
ifeq ($(wildcard $(PROTOC_BIN)),$(PROTOC_BIN))
PROTOC := $(PROTOC_BIN)
endif

.PHONY: protos clean-protos check-protos protoc29

# Download the pinned protoc 29 for this OS/arch into .tooling/ (one-time, no
# Homebrew, no PATH changes). Run this once, then `make protos` picks it up.
# Hermetic generation is the whole point: it stops "whatever protoc is on PATH"
# (e.g. brew's 35 → gencode 7.x) from producing *_pb2.py the pinned 5.29 runtime
# can't load. See the version-pin notes in this file's header.
protoc29:
	@os=$$(uname -s); arch=$$(uname -m); \
	case "$$os/$$arch" in \
	  Darwin/arm64)  plat=osx-aarch_64 ;; \
	  Darwin/x86_64) plat=osx-x86_64 ;; \
	  Linux/x86_64)  plat=linux-x86_64 ;; \
	  Linux/aarch64) plat=linux-aarch_64 ;; \
	  *) echo "ERROR: unsupported $$os/$$arch — download protoc $(PROTOC_VERSION) manually" >&2; exit 1 ;; \
	esac; \
	url="https://github.com/protocolbuffers/protobuf/releases/download/v$(PROTOC_VERSION)/protoc-$(PROTOC_VERSION)-$$plat.zip"; \
	echo "[protoc29] $$url -> $(PROTOC_DIR)"; \
	rm -rf "$(PROTOC_DIR)"; mkdir -p "$(PROTOC_DIR)"; \
	curl -fLsS "$$url" -o "$(PROTOC_DIR)/protoc.zip" && \
	unzip -oq "$(PROTOC_DIR)/protoc.zip" -d "$(PROTOC_DIR)" && rm -f "$(PROTOC_DIR)/protoc.zip" && \
	"$(PROTOC_BIN)" --version

protos:
	@v="$$($(PROTOC) --version 2>/dev/null)"; \
	case "$$v" in \
	  *"libprotoc 29."*) : ;; \
	  *) echo "ERROR: got '$$v' — this repo requires libprotoc 29.x so the committed" >&2; \
	     echo "       gencode matches the pinned protobuf 5.29 runtime. A newer protoc" >&2; \
	     echo "       (e.g. brew's 35 → gencode 7.x) produces *_pb2.py that the 5.29" >&2; \
	     echo "       runtime in CI/deploy REFUSES to load (VersionError). Run" >&2; \
	     echo "       'make protoc29' once to bootstrap the pinned protoc, then re-run" >&2; \
	     echo "       'make protos' (or pass PROTOC=/path/to/protoc29/bin/protoc)." >&2; \
	     exit 1 ;; \
	esac
	$(PROTOC) -I msgs --python_out=$(PY_OUT) $(PROTOS)
	touch $(PY_OUT)/__init__.py

# CI guard: regenerating must produce no diff (proto and committed pb2 in sync).
check-protos: protos
	git diff --exit-code $(PY_OUT) || (echo "ERROR: *_pb2.py out of sync with msgs/*.proto — run 'make protos'"; exit 1)

clean-protos:
	rm -f $(PY_OUT)/*_pb2.py
