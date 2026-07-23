# 通用 GitHub 上传工具

`easy_publish.py` 将任意目录复制到临时 Git 仓库后发布，不修改源目录的文件、
Git 配置或提交历史。文件清单由 Git 生成，因此自动遵循源目录中的 `.gitignore`。

## 准备

1. 安装 Python 3.10+、Git 和 [GitHub CLI](https://cli.github.com/)。
2. 登录一次：`gh auth login`。
3. 上传超过 100 MiB 的文件时安装 Git LFS。

## 使用

先预览实际上传内容：

```bash
python3 tools/easy_publish.py /path/to/project owner/repository --dry-run
```

上传到已有空仓库：

```bash
python3 tools/easy_publish.py /path/to/project owner/repository
```

目标仓库不存在时创建私有仓库：

```bash
python3 tools/easy_publish.py /path/to/project owner/repository --create private
```

覆盖已有 `main` 分支，并通过 Git LFS 上传超限文件：

```bash
python3 tools/easy_publish.py /path/to/project owner/repository --force --lfs
```

追加不上传的目录或文件规则：

```bash
python3 tools/easy_publish.py /path/to/project owner/repository \
  --exclude 'runs/' --exclude '*.zip' --dry-run
```

默认不会强制覆盖远端历史；只有显式指定 `--force` 才会执行强制推送。上传完成后，
工具会比较本地提交与 GitHub 分支 SHA，二者一致才报告成功。
