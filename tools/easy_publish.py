#!/usr/bin/env python3
"""Publish any local directory to GitHub without modifying the source tree."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


GITHUB_LIMIT = 100 * 1024 * 1024
DEFAULT_EXCLUDES = (
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".venv/",
    "venv/",
    "node_modules/",
    "*.py[cod]",
    ".DS_Store",
    "Thumbs.db",
)
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class PublishError(RuntimeError):
    pass


def run(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise PublishError(f"命令失败: {' '.join(args)}\n{detail}")
    return result


def require(command: str, hint: str) -> None:
    if shutil.which(command) is None:
        raise PublishError(f"未找到 {command}；{hint}")


def prepare_snapshot(source: Path, excludes: list[str], temp_root: Path) -> tuple[Path, list[str]]:
    index_repo = temp_root / "index"
    stage = temp_root / "stage"
    run(["git", "init", "-q", str(index_repo)])

    git_dir = index_repo / ".git"
    exclude_file = git_dir / "info" / "exclude"
    rules = [*DEFAULT_EXCLUDES, *excludes]
    exclude_file.write_text("\n".join(rules) + "\n", encoding="utf-8")

    git = ["git", f"--git-dir={git_dir}", f"--work-tree={source}"]
    run([*git, "add", "-A"])
    listed = run([*git, "ls-files", "-z"]).stdout
    files = [name for name in listed.split("\0") if name]
    if not files:
        raise PublishError("忽略规则应用后没有可上传文件")

    stage.mkdir()
    prefix = str(stage.resolve()) + os.sep
    run([*git, "checkout-index", "-a", "-f", f"--prefix={prefix}"])
    return stage, files


def file_summary(stage: Path, files: list[str]) -> tuple[int, list[tuple[str, int]]]:
    sizes = [(name, (stage / name).lstat().st_size) for name in files]
    return sum(size for _, size in sizes), [(name, size) for name, size in sizes if size > GITHUB_LIMIT]


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def preview(files: list[str], total: int, large: list[tuple[str, int]]) -> None:
    print(f"待上传: {len(files)} 个文件，共 {human_size(total)}")
    for name in files[:20]:
        print(f"  {name}")
    if len(files) > 20:
        print(f"  ... 其余 {len(files) - 20} 个文件")
    if large:
        print("超过 GitHub 100 MiB 限制的文件:")
        for name, size in large:
            print(f"  {name} ({human_size(size)})")


def github_repo_exists(repo: str) -> bool:
    result = run(["gh", "repo", "view", repo, "--json", "nameWithOwner"], check=False)
    return result.returncode == 0


def ensure_github_repo(repo: str, visibility: str | None) -> None:
    run(["gh", "auth", "status", "--hostname", "github.com"])
    if github_repo_exists(repo):
        return
    if visibility is None:
        raise PublishError(f"仓库 {repo} 不存在或当前账号无权访问；需要建库时使用 --create")
    run(["gh", "repo", "create", repo, f"--{visibility}"])


def git_identity() -> tuple[str, str]:
    name = run(["git", "config", "--global", "user.name"], check=False).stdout.strip()
    email = run(["git", "config", "--global", "user.email"], check=False).stdout.strip()
    if name and email:
        return name, email
    account = json.loads(run(["gh", "api", "user"]).stdout)["login"]
    return name or account, email or f"{account}@users.noreply.github.com"


def authenticated_git() -> list[str]:
    return [
        "git",
        "-c",
        "credential.https://github.com.helper=",
        "-c",
        "credential.https://github.com.helper=!gh auth git-credential",
    ]


def publish(stage: Path, repo: str, branch: str, message: str, large: list[tuple[str, int]], force: bool) -> str:
    run(["git", "init", "-q", "-b", branch], cwd=stage)
    name, email = git_identity()
    run(["git", "config", "user.name", name], cwd=stage)
    run(["git", "config", "user.email", email], cwd=stage)

    if large:
        run(["git", "lfs", "install", "--local"], cwd=stage)
        for path, _ in large:
            run(["git", "lfs", "track", "--filename", path], cwd=stage)

    run(["git", "add", "-A"], cwd=stage)
    run(["git", "commit", "-q", "-m", message], cwd=stage)
    local_sha = run(["git", "rev-parse", "HEAD"], cwd=stage).stdout.strip()
    remote = f"https://github.com/{repo}.git"
    run(["git", "remote", "add", "origin", remote], cwd=stage)

    push = [*authenticated_git(), "push"]
    if force:
        push.append("--force")
    run([*push, "origin", f"HEAD:{branch}"], cwd=stage)
    remote_line = run([*authenticated_git(), "ls-remote", remote, f"refs/heads/{branch}"]).stdout.strip()
    remote_sha = remote_line.split()[0] if remote_line else ""
    if remote_sha != local_sha:
        raise PublishError(f"远端校验失败: local={local_sha}, remote={remote_sha or 'missing'}")
    return local_sha


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="安全地将任意目录发布到 GitHub")
    result.add_argument("source", type=Path, help="待上传目录")
    result.add_argument("repo", help="目标仓库，格式为 owner/name")
    result.add_argument("--branch", default="main", help="目标分支，默认 main")
    result.add_argument("--message", default="publish project", help="提交信息")
    result.add_argument("--exclude", action="append", default=[], help="追加 gitignore 规则，可重复")
    result.add_argument("--create", choices=("private", "public", "internal"), help="目标不存在时创建仓库")
    result.add_argument("--lfs", action="store_true", help="使用 Git LFS 上传超过 100 MiB 的文件")
    result.add_argument("--force", action="store_true", help="强制覆盖目标分支历史")
    result.add_argument("--dry-run", action="store_true", help="只生成并显示文件清单")
    result.add_argument("-y", "--yes", action="store_true", help="跳过上传前确认")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        require("git", "请先安装 Git")
        source = args.source.expanduser().resolve()
        if not source.is_dir():
            raise PublishError(f"源目录不存在: {source}")
        if not REPO_PATTERN.fullmatch(args.repo):
            raise PublishError("目标仓库必须使用 owner/name 格式")

        with tempfile.TemporaryDirectory(prefix="easy-publish-") as temp:
            stage, files = prepare_snapshot(source, args.exclude, Path(temp))
            total, large = file_summary(stage, files)
            preview(files, total, large)
            if large and not args.lfs:
                raise PublishError("存在超限文件；确认需要上传后安装 Git LFS 并添加 --lfs")
            if args.lfs:
                require("git-lfs", "请先安装 Git LFS")
            if args.dry_run:
                print("预览完成，未连接 GitHub，也未修改源目录")
                return 0
            if not args.yes and input(f"上传到 {args.repo}/{args.branch}？[y/N] ").strip().lower() != "y":
                print("已取消")
                return 0

            require("gh", "请安装 GitHub CLI 并运行 gh auth login")
            ensure_github_repo(args.repo, args.create)
            sha = publish(stage, args.repo, args.branch, args.message, large, args.force)
            print(f"上传并校验成功: {sha}")
            return 0
    except (PublishError, OSError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
