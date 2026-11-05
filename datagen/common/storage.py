#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Copy / list pipeline data on a local disk or any S3-compatible store."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import List


def data_root(*parts: str) -> str:
    """``$HALO_S3_ROOT`` joined with ``parts`` (raises if the root is unset)."""
    root = os.environ.get("HALO_S3_ROOT", "").rstrip("/")
    if not root:
        raise RuntimeError("HALO_S3_ROOT is not set (e.g. s3://my-bucket or /data/halo)")
    rel = "/".join(p.strip("/") for p in parts if p)
    return f"{root}/{rel}" if rel else root


def is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _local(path: str) -> str:
    return path[len("file://") :] if path.startswith("file://") else path


def _fail(cmd: list[str], err: str, rc: int = 1) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(rc, cmd, "", err)


class Storage:
    """S3 via the ``aws s3`` CLI, local paths via shutil (see module docstring)."""

    def __init__(self, binary: str = "aws"):
        self.binary = binary

    def _aws(
        self, args: list[str], check: bool = True, timeout: int | None = None, retries: int = 1
    ) -> subprocess.CompletedProcess:
        """Run `aws s3 <args>`; retry up to `retries` times on failure or timeout."""
        if shutil.which(self.binary) is None:
            raise _fail(
                [self.binary, "s3", *args],
                f"`{self.binary}` not found in PATH (pip install awscli) - needed for s3:// paths",
                127,
            )
        cmd = [self.binary, "s3", *args]
        endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
        if endpoint and "--endpoint-url" not in args:  # older awscli ignore the env var
            cmd += ["--endpoint-url", endpoint]
        for attempt in range(retries):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                if attempt == retries - 1:
                    raise
                continue
            if r.returncode == 0 or attempt == retries - 1:
                break
            time.sleep(2**attempt)
        if check and r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
        return r

    def cp(self, src: str, dst: str, *, timeout: int | None = None, retries: int = 1) -> None:
        """Copy one object/file. A `dst` ending in "/" (or an existing dir) keeps the name."""
        if is_s3(src) or is_s3(dst):
            self._aws(["cp", src, dst], timeout=timeout, retries=retries)
            return
        s, d = _local(src), _local(dst)
        if not os.path.isfile(s):
            raise _fail(["cp", src, dst], f"no such file: {src}")
        if d.endswith("/") or os.path.isdir(d):
            d = os.path.join(d, os.path.basename(s))
        os.makedirs(os.path.dirname(os.path.abspath(d)), exist_ok=True)
        shutil.copy2(s, d)

    def sync(
        self,
        src: str,
        dst: str,
        *,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
        timeout: int | None = None,
        retries: int = 1,
    ) -> None:
        """Copy files under `src` that are missing or differ (size/mtime) under `dst`."""
        filters = [("exclude", p) for p in (exclude or [])] + [("include", p) for p in (include or [])]
        if is_s3(src) or is_s3(dst):
            self._aws(["sync", src, dst, *_filter_args(filters)], timeout=timeout, retries=retries)
            return
        _local_tree_copy(_local(src), _local(dst), filters, only_changed=True)

    def rm(self, key: str, *, recursive: bool = False) -> None:
        if is_s3(key):
            self._aws(["rm", *(["--recursive"] if recursive else []), key])
            return
        p = _local(key)
        if os.path.isdir(p) and recursive:
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)

    def ls_lines(self, prefix: str, recursive: bool = False, timeout: int | None = None) -> List[str]:
        """Listing in `aws s3 ls` text format."""
        if is_s3(prefix):
            r = self._aws(["ls", *(["--recursive"] if recursive else []), prefix], timeout=timeout)
            return r.stdout.splitlines()
        lines = _local_ls(_local(prefix), recursive)
        if not lines:
            raise _fail(["ls", prefix], f"nothing at {prefix}")
        return lines

    def ls(self, prefix: str, *, timeout: int | None = None) -> List[str]:
        """Sorted last-component names under `prefix` (sub-prefixes without the "/")."""
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        out = []
        for line in self.ls_lines(prefix, timeout=timeout):
            parts = line.split()
            if len(parts) == 2 and parts[0] == "PRE":
                out.append(parts[1].rstrip("/"))  # "PRE name/" (sub-prefix)
            elif len(parts) >= 4:
                out.append(parts[-1])  # "DATE TIME SIZE name" (object)
        return sorted(out)

    def exists(self, key: str, *, timeout: int | None = None) -> bool:
        """True if `key` is an existing object, or a prefix with at least one entry."""
        if not is_s3(key):
            p = _local(key)
            return os.path.isfile(p) or (os.path.isdir(p) and bool(os.listdir(p)))
        if not key.endswith("/"):
            r = self._aws(["ls", key], check=False, timeout=timeout)
            name = key.rsplit("/", 1)[-1]
            if r.returncode == 0 and any(l.split()[-1] == name for l in r.stdout.splitlines() if l.strip()):
                return True
        try:
            return bool(self.ls(key, timeout=timeout))
        except subprocess.CalledProcessError:
            return False


def _filter_args(filters: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for kind, pat in filters:
        out += [f"--{kind}", pat]
    return out


def _included(rel: str, filters: list[tuple[str, str]]) -> bool:
    """`aws s3` filter semantics: everything is included, later filters win."""
    keep = True
    for kind, pat in filters:
        if fnmatch.fnmatchcase(rel, pat):
            keep = kind == "include"
    return keep


def _local_tree_copy(src: str, dst: str, filters: list[tuple[str, str]], only_changed: bool) -> None:
    if not os.path.isdir(src):
        if only_changed:  # `aws s3 sync` of a missing prefix copies nothing
            return
        raise _fail(["cp", "--recursive", src, dst], f"no such directory: {src}")
    for dirpath, _dirs, files in os.walk(src):
        for fn in files:
            s = os.path.join(dirpath, fn)
            rel = os.path.relpath(s, src).replace(os.sep, "/")
            if not _included(rel, filters):
                continue
            d = os.path.join(dst, rel)
            if only_changed and os.path.isfile(d):
                ss, ds = os.stat(s), os.stat(d)
                if ss.st_size == ds.st_size and int(ss.st_mtime) <= int(ds.st_mtime):
                    continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def _stamp(path: str) -> str:
    st = os.stat(path)
    return f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S} {st.st_size:>10}"


def _local_ls(path: str, recursive: bool) -> list[str]:
    if recursive:
        base = path.rstrip("/") if path.endswith("/") else path
        if os.path.isfile(base):
            return [f"{_stamp(base)} {os.path.abspath(base).lstrip('/')}"]
        if not os.path.isdir(base):
            return []
        out = []
        for dirpath, _dirs, files in os.walk(base):
            for fn in sorted(files):
                p = os.path.join(dirpath, fn)
                out.append(f"{_stamp(p)} {os.path.abspath(p).lstrip('/')}")
        return sorted(out, key=lambda l: l.split(None, 3)[-1])
    if path.endswith("/"):
        d, stem = path, ""
    else:
        d, stem = os.path.dirname(path) or ".", os.path.basename(path)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if not name.startswith(stem):
            continue
        p = os.path.join(d, name)
        out.append(f"{'PRE':>30} {name}/" if os.path.isdir(p) else f"{_stamp(p)} {name}")
    return out


_DEFAULT: Storage | None = None


def s3() -> Storage:
    """Process-wide Storage, created on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Storage()
    return _DEFAULT


_IGNORED_FLAGS = {"--only-show-errors", "--quiet", "--no-progress", "--size-only"}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "root":
        print(data_root(*rest))
        return 0
    paths = [a for a in rest if not a.startswith("--")]
    if cmd != "exists" and any(is_s3(a) for a in rest):
        r = subprocess.run(
            [s3().binary, "s3", *argv]
            + (
                ["--endpoint-url", os.environ["AWS_ENDPOINT_URL"]]
                if os.environ.get("AWS_ENDPOINT_URL") and "--endpoint-url" not in argv
                else []
            )
        )
        return r.returncode
    filters, recursive, pos, i = [], False, [], 0
    while i < len(rest):
        a = rest[i]
        if a in ("--exclude", "--include"):
            filters.append((a[2:], rest[i + 1]))
            i += 2
            continue
        if a in ("--recursive", "-r"):
            recursive = True
        elif a in _IGNORED_FLAGS:
            pass
        elif a.startswith("--"):
            print(f"storage: unsupported flag for local paths: {a}", file=sys.stderr)
            return 2
        else:
            pos.append(a)
        i += 1
    st = s3()
    try:
        if cmd == "cp":
            src, dst = pos
            if recursive:
                _local_tree_copy(_local(src), _local(dst), filters, only_changed=False)
            else:
                st.cp(src, dst)
        elif cmd == "sync":
            src, dst = pos
            _local_tree_copy(_local(src), _local(dst), filters, only_changed=True)
        elif cmd == "ls":
            for line in st.ls_lines(pos[0], recursive=recursive):
                print(line)
        elif cmd == "exists":
            return 0 if st.exists(paths[0]) else 1
        elif cmd == "rm":
            st.rm(pos[0], recursive=recursive)
        else:
            print(f"storage: unknown command {cmd!r}", file=sys.stderr)
            return 2
    except subprocess.CalledProcessError as e:
        print(f"storage: {e.stderr or e}", file=sys.stderr)
        return e.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
