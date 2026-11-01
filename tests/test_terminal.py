#!/usr/bin/env python3
"""Exercise the shared terminal without loading a model or installing packages."""
import fcntl
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import tempfile
import termios
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TerminalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.directory.name) / "terminal"
        subprocess.run([
            "cc", "-std=c11", "-D_GNU_SOURCE", "-ffunction-sections", "-fdata-sections",
            "-Wl,--gc-sections", "-pthread", "-I", str(ROOT),
            str(ROOT / "tests/loina_terminal_driver.c"), str(ROOT / "loina_terminal.c"),
            str(ROOT / "linenoise.c"), str(ROOT / "loina_renderer.c"), "-o", str(cls.binary),
        ], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.start()

    def start(self, **extra):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 120, 0, 0))
        self.process = subprocess.Popen([self.binary], stdin=slave, stdout=slave, stderr=slave,
                                       env=dict(os.environ, HOME=self.home.name, TERM="xterm-256color", **extra))
        os.close(slave)
        self.output = b""
        self.collect(.2)

    def restart(self, **extra):
        self.process.kill()
        self.process.wait()
        os.close(self.master)
        self.start(**extra)

    def tearDown(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        os.close(self.master)
        self.home.cleanup()

    def collect(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if select.select([self.master], [], [], .03)[0]:
                try:
                    self.output += os.read(self.master, 65536)
                except OSError:
                    break

    def send(self, data, seconds=.2):
        os.write(self.master, data)
        self.collect(seconds)

    def finish(self):
        self.send(b"/quit\r")
        self.assertEqual(self.process.wait(timeout=2), 0)
        self.assertIn(b"\x1b[r", self.output)

    def transcript(self):
        payload = b"".join(re.findall(rb"\x1b8(.*?)\x1b7", self.output, re.S))
        return re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", payload)

    def test_eof_save_confirmation(self):
        for answer in (b"y", b"n"):
            self.restart(LOINA_TEST_EXIT_SAVE="1")
            self.send(b"hello\r",1.5)
            self.send(b"\x04")
            self.assertIsNone(self.process.poll())
            self.assertIn(b"Save current session? (y/n) ",self.output)
            self.assertTrue(termios.tcgetattr(self.master)[3] & termios.ECHO)
            self.send(answer+b"\r")
            self.assertEqual(self.process.wait(timeout=2),0)
            self.assertEqual(b"CHECKPOINT_WRITTEN" in self.output,answer==b"y")
            self.assertEqual(b"saved session fixture (chat + KV)" in self.output,answer==b"y")

    def test_eof_skips_clean_sessions(self):
        self.restart(LOINA_TEST_EXIT_SAVE="1")
        self.send(b"\x04")
        self.assertEqual(self.process.wait(timeout=2),0)
        self.assertNotIn(b"Save current session?",self.output)
        self.restart(LOINA_TEST_EXIT_SAVE="1")
        self.send(b"hello\r",1.5)
        self.send(b"/save\r")
        self.send(b"\x04")
        self.assertEqual(self.process.wait(timeout=2),0)
        self.assertNotIn(b"Save current session?",self.output)

    def test_failed_save_can_return_to_chat(self):
        self.restart(LOINA_TEST_EXIT_SAVE="1",LOINA_TEST_SAVE_FAIL="1")
        self.send(b"hello\r",1.5)
        self.send(b"\x04")
        self.send(b"y\r")
        self.assertIn(b"save failed: cannot write KV fixture",self.output)
        self.assertIn(b"Continue anyway? (y/n) ",self.output)
        self.send(b"n\r")
        self.assertIsNone(self.process.poll())
        self.assertIn(b"loina-agent> ",self.output[-2000:])
        self.send(b"\x04")
        self.send(b"n\r")
        self.assertEqual(self.process.wait(timeout=2),0)

    def test_screen_queue_and_resize(self):
        self.assertIn(b"\x1b[2J", self.output)
        self.assertIn(b"\x1b[1;22r", self.output)
        self.assertLess(self.output.index(b"\x1b[1;22r"), self.output.index(b"Loina"))
        self.send(b"first\r")
        self.send(b"second\r")
        self.assertIn(b"queued: second", self.output)
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        self.collect(2.5)
        self.assertIn(b"\x1b[1;28r", self.output)
        self.assertIn(b"generation 3 tokens 12.5 t/s", self.output)
        self.assertIn(b"*\x1b[1;97m second", self.output)
        self.finish()

    def test_queue_edit_interrupt_and_eof(self):
        self.send(b"first\r")
        self.send(b"second\r")
        self.send(b"\x18")
        self.assertIn(b"loina-agent> second", self.output)
        self.send(b"\x15edited\r")
        self.assertIn(b"queued: edited", self.output)
        self.send(b"\x1b", 1.6)
        self.assertIn(b"*\x1b[1;97m edited", self.output)
        self.send(b"\x04")
        self.assertEqual(self.process.wait(timeout=2), 0)
        self.assertTrue(termios.tcgetattr(self.master)[3] & termios.ECHO)

    def test_cancel_reaches_backend_before_tokens(self):
        self.send(b"prefill\r", .2)
        self.send(b"\x03", .4)
        self.assertIn(b"BACKEND_CANCELLED_BEFORE_TOKEN", self.transcript())
        self.assertIn(b"idle", self.output)
        self.finish()

    def test_output_is_live_before_cancel(self):
        self.send(b"burst\r", 1.5)
        self.assertIsNone(self.process.poll())
        self.assertIn(b"LIVE_BEFORE_CANCEL", self.output)
        self.assertNotIn(b"TTY_NONBLOCK", self.output)
        payload = b"".join(re.findall(rb"\x1b8(.*?)\x1b7", self.output, re.S))
        for i in range(1200):
            self.assertIn(("live tool output %04d" % i).encode(), payload,
                          "missing output line %d" % i)
        self.assertEqual(self.output.count(b"\x1b[?2026h"), self.output.count(b"\x1b[?2026l"))
        self.send(b"\x03", .4)
        self.assertIn(b"CANCELLED_WORKER", self.output)
        self.assertIn(b"idle", self.output)
        self.finish()

    def test_tool_output_after_thinking_is_live(self):
        self.send(b"think-tool\r", .2)
        self.assertIn(b"thinking text", self.transcript())
        self.assertIn(b"preparing tool call", self.output)
        self.collect(.7)
        plain = self.transcript()
        self.assertIn(b"CODE_VISIBLE", plain)
        self.assertIn("🛠️ write ".encode(), plain)
        self.assertIn(b"path=mini_redis.py", plain)
        self.assertIn(b"if 1 < 2:", plain)
        self.assertNotIn(b"<tool_call>", plain)
        self.assertIsNone(self.process.poll())
        self.assertNotIn(b"CANCELLED_WORKER", self.output)
        self.send(b"\x03", .4)
        self.assertIn(b"CANCELLED_WORKER", self.output)
        self.finish()

    def test_unicode_paste_and_draft(self):
        self.send("hélo".encode() + b"\x1b[Dl\r")
        self.send(b"draft", 1.5)
        self.assertIn("*\x1b[1;97m héllo".encode(), self.output)
        self.assertIn(b"loina-agent> draft", self.output)
        self.send(b"\x15\x1b[200~alpha\nbeta\x1b[201~\r", 1.5)
        self.assertIn(b"alpha\r\nbeta", self.output)
        self.finish()


if __name__ == "__main__":
    unittest.main()
