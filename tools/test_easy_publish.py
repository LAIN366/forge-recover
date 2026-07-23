import tempfile
import unittest
from pathlib import Path

import easy_publish


class EasyPublishTests(unittest.TestCase):
    def test_snapshot_honors_gitignore_defaults_and_extra_rules(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            (source / ".gitignore").write_text("*.log\n", encoding="utf-8")
            (source / "keep.txt").write_text("keep", encoding="utf-8")
            (source / "ignored.log").write_text("log", encoding="utf-8")
            (source / "extra.bin").write_bytes(b"skip")
            cache = source / "__pycache__"
            cache.mkdir()
            (cache / "module.pyc").write_bytes(b"cache")

            temp_root = root / "temp"
            temp_root.mkdir()
            stage, files = easy_publish.prepare_snapshot(source, ["extra.bin"], temp_root)

            self.assertEqual(files, [".gitignore", "keep.txt"])
            self.assertTrue((stage / "keep.txt").is_file())
            self.assertFalse((stage / "ignored.log").exists())
            self.assertFalse((stage / "extra.bin").exists())

    def test_file_summary_detects_github_limit(self):
        with tempfile.TemporaryDirectory() as root:
            stage = Path(root)
            (stage / "small.bin").write_bytes(b"x")
            large = stage / "large.bin"
            with large.open("wb") as stream:
                stream.truncate(easy_publish.GITHUB_LIMIT + 1)

            total, oversized = easy_publish.file_summary(stage, ["small.bin", "large.bin"])

            self.assertEqual(total, easy_publish.GITHUB_LIMIT + 2)
            self.assertEqual(oversized, [("large.bin", easy_publish.GITHUB_LIMIT + 1)])

    def test_repository_name_validation(self):
        self.assertIsNotNone(easy_publish.REPO_PATTERN.fullmatch("owner/repo-name"))
        self.assertIsNone(easy_publish.REPO_PATTERN.fullmatch("https://github.com/owner/repo"))


if __name__ == "__main__":
    unittest.main()
