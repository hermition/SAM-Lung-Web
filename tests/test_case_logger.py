import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from web_ui.case_logger import CASE_ID_PATTERN, CaseLogger


class CaseLoggerTest(unittest.TestCase):
    def test_complete_case_audit_log(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "cases"
            logger = CaseLogger(output_root, {"backend": "fake", "checkpoint": "model.pt"})
            image = np.zeros((12, 16, 3), dtype=np.uint8)
            image[:, :, 1] = 90
            session_id = "doctor-browser-session"

            case_id = logger.start_case(image, session_id)
            first_mask = np.zeros((12, 16), dtype=bool)
            first_mask[2:6, 3:8] = True
            first_point = (4.5, 3.0, 1)
            logger.record_interaction(
                case_id,
                "point",
                [first_point],
                mask=first_mask,
                score=0.8,
                point=first_point,
            )
            second_point = (10.0, 9.0, 0)
            second_mask = first_mask.copy()
            second_mask[5, 7] = False
            logger.record_interaction(
                case_id,
                "point",
                [first_point, second_point],
                mask=second_mask,
                score=0.9,
                point=second_point,
            )
            logger.record_interaction(case_id, "undo", [first_point], mask=first_mask, score=0.8)
            logger.save_final(case_id, first_mask, image)

            self.assertRegex(case_id, CASE_ID_PATTERN)
            case_dir = output_root / case_id
            manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "saved")
            self.assertEqual(
                manifest["session_hash"], hashlib.sha256(session_id.encode("utf-8")).hexdigest()
            )
            self.assertNotIn(session_id, (case_dir / "case.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["interactions"]), 3)
            self.assertEqual(manifest["interactions"][0]["prompt"]["label_name"], "foreground")
            self.assertEqual(manifest["interactions"][1]["prompt"]["label_name"], "background")
            self.assertEqual(manifest["interactions"][1]["active_prompts"][0]["x"], 4.5)
            for interaction in manifest["interactions"]:
                self.assertTrue(interaction["recorded_at"].endswith("Z"))
                self.assertTrue(interaction["completed_at"].endswith("Z"))
                self.assertTrue((case_dir / interaction["mask_path"]).is_file())
            with Image.open(case_dir / "input.png") as saved_input:
                self.assertEqual(saved_input.size, (16, 12))
            self.assertTrue((case_dir / "final_mask.png").is_file())
            self.assertTrue((case_dir / "final_overlay.png").is_file())
            self.assertEqual(os.stat(case_dir / "case.json").st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
