import json
import http.client
import tempfile
import threading
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import app


class PlannerTest(unittest.TestCase):
    def test_api_validation_errors_are_json(self):
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request("POST", "/api/plan", "{}", {"Content-Type": "text/plain"})
            response = connection.getresponse()
            data = json.loads(response.read())
            connection.close()
        finally:
            thread.join(timeout=2)
            server.server_close()

        self.assertEqual(response.status, 415)
        self.assertIn("application/json", response.getheader("Content-Type"))
        self.assertIn("application/json", data["error"])

    def test_calendar_handles_non_json_responses_and_refreshes(self):
        calendar = (app.ROOT / "docs/trip-calendar.html").read_text()
        self.assertIn("const responseBody = await response.text()", calendar)
        self.assertIn("result = JSON.parse(responseBody)", calendar)
        self.assertIn("Refreshing your trip with the new details", calendar)
        self.assertIn("selectedKey = '';", calendar)

    def test_accepts_thirty_day_plan_without_daily_limit(self):
        itinerary = {"title": "Rome", "summary": "Estimates only", "days": [
            {"date": f"2027-01-{i:02d}", "city": "Rome", "activities": []} for i in range(1, 31)
        ]}
        response = {"choices": [{"message": {"content": json.dumps(itinerary)}}]}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return json.dumps(response).encode()

        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder) / "key"
            key.write_text("test-key")
            with patch.dict(app.os.environ, {"OPENROUTER_KEY_FILE": str(key)}), \
                 patch.object(app, "urlopen", return_value=FakeResponse()) as upstream:
                self.assertEqual(app.plan({"prompt": "Rome January 1–30, 2027"}), itinerary)
                payload = json.loads(upstream.call_args.args[0].data)
                self.assertEqual(payload["model"], "deepseek/deepseek-v4.1-flash")
                self.assertEqual(payload["reasoning"]["effort"], "high")
                self.assertNotIn("max_tokens", payload)
                self.assertEqual(payload["plugins"][0]["max_results"], 10)
                self.assertEqual(app.plan({"prompt": "Another trip request"}), itinerary)
                self.assertEqual(upstream.call_count, 2)

    def test_multiple_revisions_use_latest_plan(self):
        itinerary = {"title": "Rome", "summary": "Estimates only", "days": [
            {"date": f"2027-04-0{i}", "city": "Rome", "activities": []} for i in range(5, 8)
        ]}
        response = {"choices": [{"message": {"content": json.dumps(itinerary)}}]}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return json.dumps(response).encode()

        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder) / "key"
            key.write_text("test-key")
            with patch.dict(app.os.environ, {"OPENROUTER_KEY_FILE": str(key)}), \
                 patch.object(app, "urlopen", return_value=FakeResponse()) as upstream:
                first = app.plan({"prompt": "Rome April 5–7, 2027"})
                second = app.plan({"prompt": "Add more museums", "previous": first})
                app.plan({"prompt": "Make day two less busy", "previous": second})

                self.assertEqual(upstream.call_count, 3)
                last_payload = json.loads(upstream.call_args.args[0].data)
                last_prompt = last_payload["messages"][1]["content"]
                self.assertIn("Previous itinerary:", last_prompt)
                self.assertIn("Revision requested: Make day two less busy", last_prompt)

    def test_accepts_every_day_in_a_fifteen_day_trip(self):
        start = date(2027, 4, 5)
        itinerary = {"title": "Italy", "summary": "15-day draft", "days": [
            {"date": (start + timedelta(days=index)).isoformat(), "city": "Rome", "activities": []}
            for index in range(15)
        ]}
        response = {"choices": [{"message": {"content": json.dumps(itinerary)}}]}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return json.dumps(response).encode()

        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder) / "key"
            key.write_text("test-key")
            with patch.dict(app.os.environ, {"OPENROUTER_KEY_FILE": str(key)}), \
                 patch.object(app, "urlopen", return_value=FakeResponse()):
                result = app.plan({"prompt": "Plan Rome from April 5 through April 19, 2027"})

        self.assertEqual(len(result["days"]), 15)
        self.assertEqual(result["days"][0]["date"], "2027-04-05")
        self.assertEqual(result["days"][-1]["date"], "2027-04-19")

    def test_rejects_more_than_thirty_days(self):
        itinerary = {"title": "Long trip", "summary": "Too long", "days": [
            {"date": f"2027-01-{i:02d}", "city": "Rome", "activities": []} for i in range(1, 32)
        ]}
        response = {"choices": [{"message": {"content": json.dumps(itinerary)}}]}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return json.dumps(response).encode()

        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder) / "key"
            key.write_text("test-key")
            with patch.dict(app.os.environ, {"OPENROUTER_KEY_FILE": str(key)}), \
                 patch.object(app, "urlopen", return_value=FakeResponse()):
                with self.assertRaisesRegex(RuntimeError, "incomplete itinerary"):
                    app.plan({"prompt": "Rome January 1–31, 2027"})


if __name__ == "__main__":
    unittest.main()
