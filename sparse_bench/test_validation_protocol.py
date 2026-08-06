import unittest
from validation_protocol import hold_out_validation_targets


class ValidationProtocolTests(unittest.TestCase):
    def test_removes_only_verified_source_target(self):
        sessions = {"hid_train_1": [1, 2, 3], "amazon": [4, 5]}
        validation = {
            "hid_train_1_v": {"context": [1, 2], "targets": [3]},
            "amazon": {"context": [4, 5], "targets": [6]},
        }
        out = hold_out_validation_targets(sessions, validation)
        self.assertEqual(out["hid_train_1"], [1, 2])
        self.assertEqual(out["amazon"], [4, 5])
        self.assertEqual(sessions["hid_train_1"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
