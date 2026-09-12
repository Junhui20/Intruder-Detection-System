"""Event captions: talk to Ollama when it is there, stay quiet when it is not."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests

sys.path.append(str(Path(__file__).parent.parent))

from core.event_captions import EventCaptioner

FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def tags(*names):
    reply = mock.Mock(ok=True)
    reply.json.return_value = {"models": [{"name": n} for n in names]}
    return reply


class TestEventCaptioner(unittest.TestCase):
    def test_off_when_ollama_is_unreachable(self):
        with mock.patch(
            "core.event_captions.requests.get", side_effect=requests.ConnectionError
        ):
            captioner = EventCaptioner()
        self.assertFalse(captioner.enabled)
        self.assertIsNone(captioner.describe(FRAME))

    def test_off_when_model_is_not_pulled(self):
        with mock.patch(
            "core.event_captions.requests.get", return_value=tags("llama3:latest")
        ):
            self.assertFalse(EventCaptioner("qwen2.5vl:3b").enabled)

    def test_tag_suffix_counts_as_the_model_and_it_is_kept_loaded(self):
        with mock.patch(
            "core.event_captions.requests.get", return_value=tags("qwen2.5vl:latest")
        ), mock.patch("core.event_captions.requests.post") as post:
            self.assertTrue(EventCaptioner("qwen2.5vl").enabled)
        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], -1)

    def test_describe_sends_the_frame_and_keeps_one_line(self):
        with mock.patch(
            "core.event_captions.requests.get", return_value=tags("qwen2.5vl:3b")
        ), mock.patch("core.event_captions.requests.post"):
            captioner = EventCaptioner("qwen2.5vl:3b")
        reply = mock.Mock()
        reply.json.return_value = {
            "response": " A man in a red jacket at the gate.\nHe is holding a bag."
        }
        with mock.patch(
            "core.event_captions.requests.post", return_value=reply
        ) as post:
            self.assertEqual(
                captioner.describe(FRAME, "person"),
                "A man in a red jacket at the gate.",
            )
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "qwen2.5vl:3b")
        self.assertIn("person", body["prompt"])
        self.assertEqual(len(body["images"]), 1)
        self.assertFalse(body["stream"])

    def test_timeout_yields_no_caption_not_an_exception(self):
        with mock.patch(
            "core.event_captions.requests.get", return_value=tags("qwen2.5vl:3b")
        ), mock.patch("core.event_captions.requests.post"):
            captioner = EventCaptioner("qwen2.5vl:3b", timeout=1)
        with mock.patch(
            "core.event_captions.requests.post", side_effect=requests.Timeout
        ):
            self.assertIsNone(captioner.describe(FRAME))


if __name__ == "__main__":
    unittest.main()
