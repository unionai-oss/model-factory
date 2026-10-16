"""A small shell reader: what does a command actually *do*?

The permission oracle (`policy.py`) never matches on raw command text, because
raw text is exactly what an obfuscated command is designed to defeat. It asks
this module for the command's **effects** — which paths it reads, writes or
destroys, whether it runs code, touches the network, escalates privilege, and
so on — and applies its rules to those.

What the reader undoes, because each is a real way past a regex gate:

- quoting and escapes:  ``r''m``, ``"rm"``, ``\\rm``  ->  ``rm``
- chaining:             ``a && b``, ``a; b``, ``a || b``, ``a | b``, ``a & b``
- nesting:              ``$(...)``, backticks, ``<(...)``, ``bash -c '...'``,
                        ``sh -c "..."``, ``eval "..."``, ``( ... )`` groups
- wrappers:             ``env X=1 cmd``, ``nohup``, ``time``, ``xargs cmd``,
                        ``command``, ``exec``, ``timeout 10 cmd``
- home expansion:       ``~``, ``$HOME``, ``${HOME}``
- relative paths:       ``..`` is resolved against the cwd, and ``cd`` earlier
                        in the same script moves the cwd for what follows

What it does NOT do, and says so: it has no filesystem, so it cannot follow a
symlink (the file tools carry a harness-resolved path for that), and it cannot
know the value of an arbitrary ``$VAR`` — a path built from one is
*unresolvable*, and the policy treats an unresolvable delete target as the
worst case, because ``rm -rf "$BUILD_DIR/"`` with an empty variable is
``rm -rf /``.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

# ── effects ─────────────────────────────────────────────────────────────

READ = "read"  # reads a path
WRITE = "write"  # creates/modifies a path
DESTROY = "destroy"  # deletes/overwrites/re-permissions a path
EXEC = "exec"  # runs project code (tests, builds, scripts)
INSTALL = "install"  # installs packages
NETWORK = "network"  # talks to a remote host
PUSH = "push"  # git push
GIT_DISCARD = "git_discard"  # throws away work in the repo
SECRET = "secret"  # reads a credential file or secret env var
PRIVILEGE = "privilege"  # sudo & friends
PIPE_TO_SHELL = "pipe_to_shell"  # downloaded code fed to an interpreter
PERSIST = "persist"  # writes a startup file / ssh config / crontab / git hook
METADATA = "metadata"  # cloud instance-metadata endpoint
UNKNOWN = "unknown"  # a command the reader does not recognise


@dataclass(frozen=True)
class Effect:
    kind: str
    #: Normalized absolute path, '' when not a path effect, None when the
    #: path could not be resolved (an unknown $VAR, or a substitution).
    path: str | None = ""
    detail: str = ""
    recursive: bool = False
    force: bool = False
    branch: str = ""


@dataclass
class Context:
    """What the reader needs to resolve paths."""

    workspace_root: str
    cwd: str
    home: str
    git_branch: str = ""


# ── lexing ──────────────────────────────────────────────────────────────

#: Placeholder for a substitution inside a word; its body is analysed
#: separately, and a path containing one is unresolvable.
SUB = "\x00SUB\x00"


@dataclass
class Command:
    argv: list[str] = field(default_factory=list)
    redirects: list[tuple[str, str]] = field(default_factory=list)
    subs: list[str] = field(default_factory=list)


@dataclass
class Pipeline:
    commands: list[Command] = field(default_factory=list)


def _match_paren(s: str, i: int) -> int:
    """Index just past the ')' closing the '(' at s[i-1]. Quote-aware."""
    depth = 1
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "'":
            j = s.find("'", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == '"':
            i += 1
            while i < n and s[i] != '"':
                i += 2 if s[i] == "\\" else 1
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def parse(script: str) -> list[Pipeline]:
    """Split a script into pipelines of simple commands.

    ``a && b | c; d`` -> [[a], [b, c], [d]]. Separators other than ``|`` all
    end a pipeline: for permission purposes ``&&``, ``||``, ``;`` and ``&``
    mean the same thing — every part may run.
    """
    pipelines: list[Pipeline] = []
    pipe = Pipeline()
    cmd = Command()
    word: list[str] = []
    in_word = False
    pending_redirect: str | None = None
    i, n = 0, len(script)

    def end_word():
        nonlocal word, in_word, pending_redirect
        if in_word:
            w = "".join(word)
            if pending_redirect is not None:
                cmd.redirects.append((pending_redirect, w))
                pending_redirect = None
            else:
                cmd.argv.append(w)
        word, in_word = [], False

    def end_command():
        nonlocal cmd
        end_word()
        if cmd.argv or cmd.redirects or cmd.subs:
            pipe.commands.append(cmd)
        cmd = Command()

    def end_pipeline():
        nonlocal pipe
        end_command()
        if pipe.commands:
            pipelines.append(pipe)
        pipe = Pipeline()

    while i < n:
        c = script[i]
        if c == "\\":
            if i + 1 < n and script[i + 1] == "\n":
                i += 2  # line continuation
                continue
            if i + 1 < n:
                word.append(script[i + 1])
                in_word = True
            i += 2
            continue
        if c == "'":
            j = script.find("'", i + 1)
            j = n if j < 0 else j
            word.append(script[i + 1 : j])
            in_word = True
            i = j + 1
            continue
        if c == '"':
            in_word = True
            i += 1
            while i < n and script[i] != '"':
                ch = script[i]
                if ch == "\\" and i + 1 < n and script[i + 1] in '"\\$`':
                    word.append(script[i + 1])
                    i += 2
                    continue
                if ch == "$" and i + 1 < n and script[i + 1] == "(":
                    end = _match_paren(script, i + 2)
                    cmd.subs.append(script[i + 2 : end - 1])
                    word.append(SUB)
                    i = end
                    continue
                if ch == "`":
                    j = script.find("`", i + 1)
                    j = n if j < 0 else j
                    cmd.subs.append(script[i + 1 : j])
                    word.append(SUB)
                    i = j + 1
                    continue
                word.append(ch)
                i += 1
            i += 1
            continue
        if c == "$" and i + 1 < n and script[i + 1] == "(":
            end = _match_paren(script, i + 2)
            cmd.subs.append(script[i + 2 : end - 1])
            word.append(SUB)
            in_word = True
            i = end
            continue
        if c == "`":
            j = script.find("`", i + 1)
            j = n if j < 0 else j
            cmd.subs.append(script[i + 1 : j])
            word.append(SUB)
            in_word = True
            i = j + 1
            continue
        if c in "<>" and i + 1 < n and script[i + 1] == "(":
            # process substitution: <(curl ...) — a word whose content runs
            end = _match_paren(script, i + 2)
            cmd.subs.append(script[i + 2 : end - 1])
            word.append(SUB)
            in_word = True
            i = end
            continue
        if c == "#" and not in_word:
            j = script.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in " \t":
            end_word()
            i += 1
            continue
        if c == "\n" or c == ";":
            end_pipeline()
            i += 1
            continue
        if c == "&":
            if script.startswith("&&", i):
                end_pipeline()
                i += 2
                continue
            if script.startswith("&>", i):
                end_word()
                pending_redirect = ">"
                i += 3 if script.startswith("&>>", i) else 2
                continue
            end_pipeline()  # background: still runs
            i += 1
            continue
        if c == "|":
            if script.startswith("||", i):
                end_pipeline()
                i += 2
                continue
            end_command()
            i += 2 if script.startswith("|&", i) else 1
            continue
        nxt = script[i + 1] if i + 1 < n else " "
        if c == ")" or (c == "(" and not in_word) or (
            c in "{}" and not in_word and nxt in " \t\n;" and not script.startswith("{}", i)
        ):
            # grouping: ( cd x && rm y ) / { a; b; } — boundaries only.
            # `{}` (find -exec's placeholder) and `${HOME}` are words.
            end_pipeline()
            i += 1
            continue
        if c in "<>":
            # an fd number glued to the operator (2>file) is not a word
            if in_word and "".join(word).isdigit():
                word, in_word = [], False
            end_word()
            op = c
            if script.startswith(">>", i):
                op = ">>"
            i += len(op)
            if i < n and script[i] == "&":
                # fd duplication (2>&1): no file involved
                i += 1
                while i < n and script[i].isdigit():
                    i += 1
                continue
            pending_redirect = op
            continue
        word.append(c)
        in_word = True
        i += 1
    end_pipeline()
    return pipelines


# ── classification tables ───────────────────────────────────────────────

INTERPRETERS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "python", "python3", "perl", "ruby", "node"}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
DOWNLOADERS = {"curl", "wget"}
NETWORK_CMDS = {"curl", "wget", "nc", "ncat", "netcat", "telnet", "ssh", "scp", "sftp", "ftp", "http", "https", "ping"}
PRIVILEGE_CMDS = {"sudo", "doas", "su", "pkexec"}

#: Read-only commands, and which of their positional args are paths.
#: "all" = every positional; "after1" = the first positional is a
#: pattern/program, the rest are paths; "none" = no path args.
READ_CMDS: dict[str, str] = {
    **dict.fromkeys(
        ["cat", "head", "tail", "less", "more", "ls", "wc", "stat", "file", "du", "tree",
         "diff", "md5sum", "sha256sum", "realpath", "readlink", "bat", "nl", "strings",
         "basename", "dirname", "cd", "pushd"],
        "all",
    ),
    **dict.fromkeys(["grep", "egrep", "fgrep", "rg", "ag", "sed", "awk", "jq", "yq", "sort", "uniq", "cut", "tr", "column"], "after1"),
    **dict.fromkeys(
        ["echo", "printf", "pwd", "which", "type", "whoami", "date", "uname", "true", "false",
         "hostname", "id", "df", "popd", "clear", "history", "test", "[", "sleep", "seq", "yes"],
        "none",
    ),
}
# sort/uniq/cut/tr read stdin or files; treating their first positional as a
# pattern is wrong for sort/uniq, so they are special-cased below.
_ALL_PATH_FILTERS = {"sort", "uniq"}

WRAPPERS = {"command", "builtin", "nohup", "time", "nice", "exec", "stdbuf", "caffeinate"}

INSTALL_PATTERNS: dict[str, set[str]] = {
    "pip": {"install"}, "pip3": {"install"}, "uv": {"add", "sync"}, "poetry": {"add", "install"},
    "npm": {"install", "i", "add", "ci"}, "pnpm": {"install", "i", "add"}, "yarn": {"add", "install"},
    "bun": {"add", "install", "i"}, "cargo": {"add", "install"}, "go": {"get", "install"},
    "brew": {"install", "upgrade"}, "apt": {"install"}, "apt-get": {"install"}, "gem": {"install"},
    "conda": {"install"}, "pipx": {"install"},
}
EXEC_CMDS = {
    "pytest", "make", "npx", "bunx", "tox", "nox", "docker", "mvn", "gradle", "tsc", "mypy",
    "ruff", "eslint", "prettier", "black", "jest", "vitest", "deno", "java", "cmake", "ninja",
    "rustc", "gcc", "clang", "just",
}
EXEC_SUBCOMMANDS: dict[str, set[str]] = {
    "npm": {"run", "test", "start", "exec", "build", "t"}, "pnpm": {"run", "test", "exec", "build", "dlx"},
    "yarn": {"run", "test", "build", "start", "dlx"}, "bun": {"run", "test", "x"},
    "cargo": {"build", "test", "run", "check", "clippy", "bench"}, "go": {"build", "test", "run", "vet", "generate"},
    "uv": {"run", "tool"}, "poetry": {"run"},
}
GIT_READ = {"status", "log", "diff", "show", "blame", "rev-parse", "ls-files", "describe", "shortlog",
            "reflog", "grep", "show-ref", "cat-file", "ls-tree", "whatchanged"}
GIT_NETWORK = {"clone", "fetch", "pull", "ls-remote", "submodule"}
PROTECTED_BRANCHES = {"main", "master"}

SECRET_BASENAMES = {
    ".netrc", ".pgpass", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "credentials",
    ".npmrc", ".pypirc", ".git-credentials", "secrets.json", "secrets.yaml", "service-account.json",
}
SECRET_DIRS = ("/.ssh/", "/.aws/", "/.gnupg/", "/.config/gcloud/", "/.kube/", "/.docker/")
NOT_SECRET_ENV = {".env.example", ".env.sample", ".env.template", ".env.dist"}
SECRET_VAR_RE = re.compile(r"\$\{?([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY)[A-Z0-9_]*)\}?")

PERSIST_HOME_FILES = (".bashrc", ".bash_profile", ".zshrc", ".zprofile", ".profile", ".zshenv",
                      ".bash_login", ".config/fish/config.fish")
PERSIST_HOME_DIRS = (".ssh/", ".config/autostart/", "Library/LaunchAgents/")

METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal", "metadata", "fd00:ec2::254", "100.100.100.200"}

_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


# ── path helpers ────────────────────────────────────────────────────────


def expand_home(word: str, home: str) -> str:
    if word == "~" or word.startswith("~/"):
        word = home + word[1:]
    return word.replace("${HOME}", home).replace("$HOME", home)


def resolve(word: str, cwd: str | None, home: str) -> str | None:
    """Absolute normalized path for a word, or None if it can't be known."""
    if SUB in word:
        return None
    w = expand_home(word, home)
    if "$" in w:
        return None  # an unknown variable
    if not w.startswith("/"):
        if cwd is None:
            return None
        w = posixpath.join(cwd, w)
    out = posixpath.normpath(w)
    return "/" + out.lstrip("/") if out.startswith("//") else out


def inside(path: str | None, root: str) -> bool:
    """Strictly inside the workspace or the root itself. ``/x/app-old`` is
    NOT inside ``/x/app`` — a plain prefix test gets that wrong."""
    if path is None:
        return False
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


def strictly_inside(path: str | None, root: str) -> bool:
    return inside(path, root) and path != root.rstrip("/")


def is_secret_path(path: str | None) -> bool:
    if not path:
        return False
    base = posixpath.basename(path.rstrip("/"))
    if base in NOT_SECRET_ENV:
        return False
    if base == ".env" or base.startswith(".env.") or base == ".envrc":
        return True
    if base in SECRET_BASENAMES or base.endswith((".pem", ".key", ".p12", ".pfx")):
        return True
    p = path if path.endswith("/") else path + "/"
    return any(d in p for d in SECRET_DIRS)


def is_persist_path(path: str | None, home: str, root: str) -> bool:
    """Files that make something run later: shell startup files, ssh config,
    autostart entries, system config, and the repo's own git hooks."""
    if not path:
        return False
    home = home.rstrip("/")
    for f in PERSIST_HOME_FILES:
        if path == f"{home}/{f}":
            return True
    for d in PERSIST_HOME_DIRS:
        if (path + "/").startswith(f"{home}/{d}"):
            return True
    if path.startswith("/etc/"):
        return True
    return (path + "/").startswith(root.rstrip("/") + "/.git/hooks/")


def url_host(url: str) -> str:
    try:
        if not _URL_RE.match(url):
            url = "http://" + url
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


# ── analysis ────────────────────────────────────────────────────────────


def _flags_and_args(argv: list[str]) -> tuple[list[str], list[str]]:
    flags, args, after_dd = [], [], False
    for a in argv:
        if after_dd:
            args.append(a)
        elif a == "--":
            after_dd = True
        elif a.startswith("-") and a != "-":
            flags.append(a)
        else:
            args.append(a)
    return flags, args


def _has_short(flags: list[str], letters: str) -> bool:
    for f in flags:
        if f.startswith("--"):
            continue
        if any(ch in f[1:] for ch in letters):
            return True
    return False


def _strip_wrappers(argv: list[str]) -> tuple[list[str], list[Effect]]:
    effects: list[Effect] = []
    while argv:
        head = posixpath.basename(argv[0])
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
            argv = argv[1:]
        elif head == "env":
            argv = argv[1:]
            while argv and (argv[0].startswith("-") or "=" in argv[0]):
                argv = argv[1:]
            if not argv:  # bare `env` dumps the environment
                return ["env"], effects
        elif head in WRAPPERS:
            argv = argv[1:]
            while argv and (argv[0].startswith("-") or re.match(r"^-?\d+$", argv[0])):
                argv = argv[1:]
        elif head == "timeout":
            argv = argv[1:]
            while argv and (argv[0].startswith("-") or re.match(r"^\d+[smhd]?$", argv[0])):
                argv = argv[1:]
        elif head == "xargs":
            argv = argv[1:]
            while argv and argv[0].startswith("-"):
                argv = argv[2:] if argv[0] in ("-I", "-n", "-P", "-L", "-d") else argv[1:]
            if not argv:
                return ["echo"], effects
            # the targets arrive on stdin: invisible to the gate
            argv = argv + [SUB]
        elif head in PRIVILEGE_CMDS:
            effects.append(Effect(PRIVILEGE, detail=head))
            argv = argv[1:]
            while argv and argv[0].startswith("-"):
                argv = argv[1:]
        else:
            break
    return argv, effects


class _Analyzer:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.cwd: str | None = ctx.cwd

    def r(self, word: str) -> str | None:
        return resolve(word, self.cwd, self.ctx.home)

    def script(self, text: str, depth: int = 0) -> list[Effect]:
        effects: list[Effect] = []
        if depth > 6:
            return [Effect(UNKNOWN, detail="nesting too deep")]
        saved = self.cwd
        for pipe in parse(text):
            effects += self.pipeline(pipe, depth)
        if depth > 0:
            self.cwd = saved  # a subshell's cd does not leak out
        return effects

    def pipeline(self, pipe: Pipeline, depth: int) -> list[Effect]:
        effects: list[Effect] = []
        downloaded = False
        for idx, cmd in enumerate(pipe.commands):
            effects += self.command(cmd, depth)
            argv, _ = _strip_wrappers(list(cmd.argv))
            head = posixpath.basename(argv[0]) if argv else ""
            if head in DOWNLOADERS:
                downloaded = True
            elif idx > 0 and downloaded and head in INTERPRETERS:
                _, args = _flags_and_args(argv[1:])
                if not args or args[0] == "-" or "-s" in argv:
                    effects.append(Effect(PIPE_TO_SHELL, detail=f"download | {head}"))
        return effects

    def command(self, cmd: Command, depth: int) -> list[Effect]:
        effects: list[Effect] = []
        for body in cmd.subs:
            effects += self.script(body, depth + 1)
        for op, target in cmd.redirects:
            p = self.r(target)
            if p in ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"):
                continue
            effects.append(Effect(READ if op == "<" else WRITE, p, detail=f"redirect {op}"))

        argv, wrap = _strip_wrappers(list(cmd.argv))
        effects += wrap
        # Secrets are a property of the WORDS, whatever command carries them:
        # `scp ~/.ssh/id_rsa host:`, `curl -F f=@.env`, `echo $GITHUB_TOKEN`.
        for w in argv[1:] if argv else []:
            for m in SECRET_VAR_RE.finditer(w):
                effects.append(Effect(SECRET, "", detail=f"${m.group(1)}"))
            cand = w.split("=@", 1)[-1] if "=@" in w else (w[1:] if w.startswith("@") else w)
            if cand and cand != "-" and SUB not in cand and not _URL_RE.match(cand):
                p = self.r(cand)
                if is_secret_path(p):
                    effects.append(Effect(SECRET, p, detail="secret file"))
        if not argv:
            return effects

        head = posixpath.basename(argv[0])
        rest = argv[1:]
        flags, args = _flags_and_args(rest)

        # interpreters given code to run inline
        if head in SHELLS and "-c" in rest:
            i = rest.index("-c")
            body = rest[i + 1] if i + 1 < len(rest) else ""
            if SUB in body:
                effects += self._sub_download(cmd, head)
            return effects + self.script(body.replace(SUB, ""), depth + 1)
        if head in ("eval", "source", "."):
            body = " ".join(rest)
            if SUB in body:
                effects += self._sub_download(cmd, head)
                if head != "eval" and not body.replace(SUB, "").strip():
                    return effects
            if head == "eval":
                return effects + self.script(body.replace(SUB, ""), depth + 1)
            return effects + [Effect(EXEC, self.r(args[0]) if args else None, detail=head)]
        if head in INTERPRETERS:
            if any(SUB in a for a in rest):
                effects += self._sub_download(cmd, head)
            if head.startswith("python") and rest[:2] == ["-m", "pip"] and "install" in rest:
                return effects + [Effect(INSTALL, detail="pip install")]
            if flags and not args and all(f in ("--version", "-V", "-v") for f in flags):
                return effects
            if not rest:
                return effects  # reads stdin; the pipeline check handles curl | sh
            return effects + [Effect(EXEC, detail=" ".join(argv[:3]))]

        if head == "env" and not rest:
            return effects + [Effect(SECRET, "", detail="dumps the environment")]
        if head == "printenv":
            if not args:
                return effects + [Effect(SECRET, "", detail="dumps the environment")]
            if any(SECRET_VAR_RE.search("$" + a) for a in args):
                return effects + [Effect(SECRET, "", detail=f"${args[0]}")]
            return effects

        if head == "git":
            return effects + self._git(rest)
        if head in ("rm", "rmdir", "unlink", "shred", "truncate"):
            recursive = head == "rm" and (_has_short(flags, "rR") or "--recursive" in flags)
            targets = args if head != "truncate" else [a for a in args if not a[:1].isdigit()]
            return effects + [Effect(DESTROY, self.r(t), detail=head, recursive=recursive) for t in targets]
        if head == "find":
            return effects + self._find(rest, depth)
        if head in ("chmod", "chown", "chgrp"):
            recursive = _has_short(flags, "R") or "--recursive" in flags
            targets = args[1:]
            if recursive:
                return effects + [Effect(DESTROY, self.r(t), detail=f"{head} -R", recursive=True) for t in targets]
            return effects + [Effect(WRITE, self.r(t), detail=head) for t in targets]
        if head == "dd":
            outs = [a[3:] for a in rest if a.startswith("of=")]
            ins = [a[3:] for a in rest if a.startswith("if=")]
            return effects + [Effect(READ, self.r(p)) for p in ins] + [
                Effect(DESTROY, self.r(p), detail="dd of=", recursive=True) for p in outs
            ]
        if head.startswith("mkfs") or head in ("fdisk", "parted", "wipefs"):
            return effects + [Effect(DESTROY, self.r(args[-1]) if args else None, detail=head, recursive=True)]
        if head == "crontab":
            if flags == ["-l"] and not args:
                return effects
            return effects + [Effect(PERSIST, "", detail="crontab")]
        if head in ("mkdir", "touch", "tee", "ln", "install", "patch"):
            if head == "patch":
                return effects + [Effect(WRITE, self.cwd, detail="patch")]
            targets = args if head != "ln" else args[-1:]
            return effects + [Effect(WRITE, self.r(t), detail=head) for t in targets]
        if head in ("cp", "rsync") and not any(":" in a and not a.startswith("/") for a in args):
            if len(args) >= 2:
                return effects + [Effect(READ, self.r(s)) for s in args[:-1]] + [
                    Effect(WRITE, self.r(args[-1]), detail=head)
                ]
            return effects + [Effect(UNKNOWN, detail=head)]
        if head == "mv":
            return effects + [Effect(WRITE, self.r(a), detail="mv") for a in args]
        if head == "sed" and (any(f.startswith("-i") or f == "--in-place" for f in flags)):
            return effects + [Effect(WRITE, self.r(a), detail="sed -i") for a in args[1:]]
        if head in ("tar", "zip", "unzip", "gzip", "gunzip"):
            if head == "tar" and not _has_short(flags, "xc") and not any(a.startswith("x") or a.startswith("c") for a in args[:1]):
                return effects + [Effect(READ, self.r(a)) for a in args[1:]]
            return effects + [Effect(WRITE, self.cwd, detail=head)] + [Effect(READ, self.r(a)) for a in args]

        if head in NETWORK_CMDS or head == "rsync" or head == "cp":
            return effects + self._network(head, rest)

        if head in INSTALL_PATTERNS and args and args[0] in INSTALL_PATTERNS[head]:
            return effects + [Effect(INSTALL, detail=f"{head} {args[0]}")]
        if head == "uv" and args[:2] == ["pip", "install"]:
            return effects + [Effect(INSTALL, detail="uv pip install")]
        if head in EXEC_SUBCOMMANDS and args and args[0] in EXEC_SUBCOMMANDS[head]:
            return effects + [Effect(EXEC, detail=f"{head} {args[0]}")]
        if head in EXEC_CMDS or argv[0].startswith("./") or argv[0].startswith("../"):
            return effects + [Effect(EXEC, detail=head)]
        if head in ("npm", "pnpm", "yarn", "cargo", "go", "uv", "pip", "node") and (
            not args or flags and all(f in ("--version", "-v", "-V") for f in flags)
        ):
            return effects

        if head in READ_CMDS:
            mode = READ_CMDS[head]
            if head in ("cd", "pushd"):
                target = args[0] if args else "~"
                self.cwd = self.r(target)
                return effects
            if mode == "none":
                return effects
            paths = args if (mode == "all" or head in _ALL_PATH_FILTERS) else args[1:]
            return effects + [Effect(READ, self.r(p)) for p in paths]
        return effects + [Effect(UNKNOWN, detail=head)]

    def _sub_download(self, cmd: Command, head: str) -> list[Effect]:
        for body in cmd.subs:
            for pipe in parse(body):
                for c in pipe.commands:
                    argv, _ = _strip_wrappers(list(c.argv))
                    if argv and posixpath.basename(argv[0]) in DOWNLOADERS:
                        return [Effect(PIPE_TO_SHELL, detail=f"{head} runs downloaded code")]
        return []

    def _network(self, head: str, rest: list[str]) -> list[Effect]:
        effects: list[Effect] = []
        hosts = []
        for i, a in enumerate(rest):
            if _URL_RE.match(a):
                hosts.append(url_host(a))
            elif "@" in a and ":" in a and not a.startswith("-"):  # user@host:path
                hosts.append(a.split("@", 1)[1].split(":", 1)[0])
            elif re.match(r"^[\w.-]+:[\w/~.-]*$", a) and head in ("scp", "rsync", "sftp"):
                hosts.append(a.split(":", 1)[0])
        if head in ("nc", "ncat", "netcat", "telnet", "ssh", "ping"):
            _, args = _flags_and_args(rest)
            if args:
                hosts.append(args[0].split("@")[-1])
        if head in ("curl", "wget"):
            # where the download lands: curl -o FILE / -O (remote name, in the
            # cwd); wget -O FILE, and by DEFAULT into the cwd unless -O -.
            out_flags = {"curl": ("-o", "--output"), "wget": ("-O", "--output-document")}[head]
            wrote = False
            for i, a in enumerate(rest):
                if a in out_flags and i + 1 < len(rest):
                    wrote = True
                    if rest[i + 1] != "-":
                        effects.append(Effect(WRITE, self.r(rest[i + 1]), detail=f"{head} {a}"))
                elif head == "wget" and a.startswith("-") and not a.startswith("--") and a.endswith("O") and i + 1 < len(rest):
                    wrote = True  # combined short flags, e.g. -qO-
                    if rest[i + 1] != "-":
                        effects.append(Effect(WRITE, self.r(rest[i + 1]), detail=f"{head} {a}"))
                elif head == "wget" and a.startswith("-") and not a.startswith("--") and "O-" in a:
                    wrote = True  # -qO- : to stdout
                elif head == "curl" and a in ("-O", "--remote-name"):
                    wrote = True
                    effects.append(Effect(WRITE, self.cwd, detail="curl -O"))
                if a in ("-T", "--upload-file") and i + 1 < len(rest):
                    effects.append(Effect(READ, self.r(rest[i + 1])))
            if head == "wget" and not wrote:
                effects.append(Effect(WRITE, self.cwd, detail="wget (saves into the cwd)"))
        for h in hosts or [""]:
            if h in METADATA_HOSTS:
                effects.append(Effect(METADATA, "", detail=h))
            effects.append(Effect(NETWORK, "", detail=h or head))
        return effects

    def _find(self, rest: list[str], depth: int) -> list[Effect]:
        roots: list[str] = []
        i = 0
        while i < len(rest) and not rest[i].startswith(("-", "(", "!")):
            roots.append(rest[i])
            i += 1
        roots = roots or ["."]
        paths = [self.r(p) for p in roots]
        effects = [Effect(READ, p) for p in paths]
        # find deletes what is UNDER its roots, so `find . -delete` at the
        # workspace root hits files inside the workspace, not the root itself.
        under = [None if p is None else posixpath.join(p, "*") for p in paths]
        if "-delete" in rest:
            effects += [Effect(DESTROY, p, detail="find -delete", recursive=True) for p in under]
        for flag in ("-exec", "-execdir", "-ok", "-okdir"):
            if flag in rest:
                j = rest.index(flag) + 1
                sub: list[str] = []
                while j < len(rest) and rest[j] not in (";", "+", "\\;"):
                    sub.append(rest[j])
                    j += 1
                for p, raw in zip(paths, roots):
                    argv = [posixpath.join(raw, "*") if a == "{}" else a for a in sub]
                    inner = self.command(Command(argv=argv), depth + 1)
                    # -exec runs once per match: a delete there is recursive
                    effects += [
                        Effect(e.kind, e.path, e.detail, True, e.force, e.branch)
                        if e.kind == DESTROY else e
                        for e in inner
                    ]
        return effects

    def _git(self, rest: list[str]) -> list[Effect]:
        # skip global options: -C <path>, -c k=v, --no-pager, ...
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in ("-C", "-c", "--git-dir", "--work-tree") else rest[1:]
        if not rest:
            return []
        sub, args = rest[0], rest[1:]
        flags, pos = _flags_and_args(args)
        if sub in GIT_READ:
            return []
        if sub == "push":
            force = any(f in ("-f", "--force", "--force-with-lease") or f.startswith("--force-with-lease=") for f in flags)
            refspecs = pos[1:] if len(pos) >= 2 else []
            branches = []
            for ref in refspecs:
                if ref.startswith("+"):
                    force = True
                    ref = ref[1:]
                dst = ref.split(":", 1)[-1]
                dst = dst.removeprefix("refs/heads/")
                branches.append(self.ctx.git_branch if dst == "HEAD" else dst)
            if not branches:
                branches = [self.ctx.git_branch]
            return [Effect(PUSH, "", detail="git push", force=force, branch=b) for b in branches] + [
                Effect(NETWORK, "", detail="git push")
            ]
        if sub in GIT_NETWORK:
            return [Effect(NETWORK, "", detail=f"git {sub}")] + (
                [Effect(WRITE, self.cwd, detail=f"git {sub}")] if sub in ("clone", "pull") else []
            )
        if sub == "clean":
            if _has_short(flags, "f") or "--force" in flags:
                return [Effect(GIT_DISCARD, self.cwd, detail="git clean -f")]
            return []  # -n: dry run
        if sub == "reset" and "--hard" in flags:
            return [Effect(GIT_DISCARD, self.cwd, detail="git reset --hard")]
        if sub in ("checkout", "restore") and ("." in pos or "--" in args) and "--staged" not in flags and not _has_short(flags, "bB"):
            return [Effect(GIT_DISCARD, self.cwd, detail=f"git {sub} (discards changes)")]
        if sub == "branch":
            if _has_short(flags, "D") or "--delete" in flags and "--force" in flags:
                return [Effect(GIT_DISCARD, self.cwd, detail="git branch -D")]
            if not pos or _has_short(flags, "alvr") and not pos:
                return []
            return [Effect(WRITE, self.cwd, detail="git branch")]
        if sub == "stash" and pos[:1] and pos[0] in ("drop", "clear"):
            return [Effect(GIT_DISCARD, self.cwd, detail=f"git stash {pos[0]}")]
        if sub == "rebase":
            return [Effect(GIT_DISCARD, self.cwd, detail="git rebase (rewrites history)")]
        if sub == "remote" and (not pos or pos[0] in ("show", "get-url")) :
            return []
        if sub == "config" and (len(pos) <= 1 or "--get" in flags or "-l" in flags or "--list" in flags):
            return []
        if sub in ("add", "commit", "checkout", "switch", "stash", "merge", "tag", "mv", "rm",
                   "restore", "cherry-pick", "revert", "init", "apply", "am", "remote", "config", "worktree"):
            return [Effect(WRITE, self.cwd, detail=f"git {sub}")]
        return [Effect(UNKNOWN, detail=f"git {sub}")]


def analyze(command: str, ctx: Context) -> list[Effect]:
    """Every effect the command may have, in order."""
    return _Analyzer(ctx).script(command)
