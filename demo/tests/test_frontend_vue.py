import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendVueSmokeTest(unittest.TestCase):
    def test_vue_sources_exist(self):
        self.assertTrue((PROJECT_ROOT / "package.json").exists())
        self.assertTrue((PROJECT_ROOT / "vite.config.js").exists())
        self.assertTrue((PROJECT_ROOT / "frontend" / "src" / "App.vue").exists())

    def test_single_modes_render_only_one_upload_slot(self):
        app_vue = (PROJECT_ROOT / "frontend" / "src" / "App.vue").read_text()
        self.assertIn('"makeup-single"', app_vue)
        self.assertIn('"fashion-single"', app_vue)
        self.assertIn("slots: 1", app_vue)
        self.assertIn('v-if="config.slots === 2"', app_vue)
        self.assertIn('v-model="aiBackend"', app_vue)
        self.assertIn('value="hf3"', app_vue)
        self.assertIn('value="aide"', app_vue)
        self.assertNotIn("mockAnalyze", app_vue)


if __name__ == "__main__":
    unittest.main()
