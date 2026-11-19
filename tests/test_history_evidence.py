import hashlib,tempfile,unittest
from pathlib import Path
from history_evidence import checkpoint_is_available,inspection_evidence_path


class HistoryEvidenceTests(unittest.TestCase):
    def test_original_and_result_are_independent_and_catalog_is_never_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['captures/inspection-757.jpg','results/result-757.jpg','results/model_references/1/clean.jpg','results/model_previews/1/clean.jpg']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'preserved bytes')
            self.assertEqual(inspection_evidence_path(root,'captures/inspection-757.jpg'),'captures/inspection-757.jpg')
            self.assertEqual(inspection_evidence_path(root,'results/result-757.jpg'),'results/result-757.jpg')
            self.assertIsNone(inspection_evidence_path(root,'captures/absent.jpg'))
            self.assertIsNone(inspection_evidence_path(root,'results/model_references/1/clean.jpg'))
            self.assertIsNone(inspection_evidence_path(root,'results/model_previews/1/clean.jpg'))
            self.assertIsNone(inspection_evidence_path(root,'../outside.jpg'))

    def test_checkpoint_must_exist_under_root_and_match_associated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'model.ckpt';p.write_bytes(b'x'*2048);expected=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertTrue(checkpoint_is_available(root,'model.ckpt',expected))
            self.assertFalse(checkpoint_is_available(root,'model.ckpt','0'*64))
            self.assertFalse(checkpoint_is_available(root,'missing.ckpt',expected))
            self.assertFalse(checkpoint_is_available(root,'../model.ckpt',expected))
            p.write_bytes(b'y'*2048)
            self.assertFalse(checkpoint_is_available(root,'model.ckpt',expected))


if __name__=='__main__':unittest.main()
