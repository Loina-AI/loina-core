#!/usr/bin/env python3
"""Check terminal storage defaults and downloads without network or model loading."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI, AGENT = (Path(p).resolve() for p in sys.argv[1:3])
sys.argv[1:] = []


class StorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.driver = Path(cls.tmp.name) / "storage"
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-I", str(ROOT),
                        str(ROOT / "tests/loina_storage_driver.c"), str(ROOT / "loina_storage.c"),
                        "-o", str(cls.driver)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        curl = self.bin / "curl"
        curl.write_text("#!" + sys.executable + "\n" + '''import json,os,pathlib,sys,time
args=sys.argv[1:]
path=pathlib.Path(args[args.index('--output')+1])
pathlib.Path(os.environ['DOWNLOAD_ARGS']).write_text(json.dumps(args))
mode=os.environ.get('DOWNLOAD_MODE','ok')
if mode=='fail': sys.exit(22)
if mode=='hold': time.sleep(30)
path.parent.mkdir(parents=True,exist_ok=True)
with path.open('ab') as f: f.write(b'abc'[path.stat().st_size:])
''')
        curl.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.root / "home"),
                        XDG_DATA_HOME=str(self.root / "ignored"),
                        LOINA_GGUF_DIR=str(self.root / "gguf"),
                        PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        DOWNLOAD_ARGS=str(self.root / "curl.json"))
        Path(self.env["HOME"]).mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_command(self, *args, **kwargs):
        return subprocess.run(args, cwd=self.root, env=self.env, text=True,
                              capture_output=True, timeout=15, **kwargs)

    def test_default_directories(self):
        self.assertEqual(self.run_command(self.driver, "cache", check=True).stdout.strip(),
                         str(Path(self.env["HOME"]) / ".loina/kvcache"))
        self.env.pop("LOINA_GGUF_DIR")
        self.assertEqual(self.run_command(self.driver, "models", check=True).stdout.strip(),
                         str(self.driver.parent / "gguf"))
        self.env["LOINA_GGUF_DIR"] = "relative models"
        self.assertEqual(self.run_command(self.driver, "models", check=True).stdout.strip(),
                         str(self.root / "relative models"))

    def test_cli_and_agent_share_paths_from_any_cwd(self):
        for binary in (CLI, AGENT):
            result = self.run_command(binary, "--models", check=True)
            self.assertIn("LFM 3 VL 2B", result.stdout)
        self.assertTrue((Path(self.env["HOME"]) / ".loina/kvcache").is_dir())
        self.assertTrue((self.root / "gguf").is_dir())
        self.assertFalse((self.root / ".loina").exists())
        self.assertFalse((self.root / "ignored").exists())
        for binary in (CLI, AGENT):
            self.run_command(binary, "--models", "--model-dir", str(self.root / "other models"),
                             "--cache-dir", str(self.root / "other cache"), check=True)
        self.assertTrue((self.root / "other models").is_dir())
        self.assertTrue((self.root / "other cache").is_dir())

    def test_resumable_fetch_and_errors(self):
        target = self.root / "model file.gguf"
        target.write_bytes(b"a")
        self.run_command(self.driver, "fetch", str(target), check=True)
        self.assertEqual(target.read_bytes(), b"abc")
        args = json.loads((self.root / "curl.json").read_text())
        self.assertEqual(args[args.index("--continue-at")+1], "-")
        self.assertEqual(args[args.index("--output")+1], str(target))
        self.env["DOWNLOAD_MODE"] = "fail"
        result = self.run_command(self.driver, "fetch", str(target))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rerun the command to resume", result.stderr)
        self.env["DOWNLOAD_MODE"] = "hold"
        result = self.run_command(self.driver, "cancel", str(target))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cancelled", result.stderr)

    def test_downloader_rejects_incomplete_models(self):
        result = self.run_command(ROOT / "download_model.sh", "chat")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Wrong artifact size", result.stderr)
        self.assertFalse(list((self.root / "gguf").glob(".loina/*.ready")))
        self.assertFalse((self.root / "gguf/LFM3-VL-2B-Q6_K_S.gguf").exists())


if __name__ == "__main__":
    unittest.main()
