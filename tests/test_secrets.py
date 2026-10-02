import tempfile
import unittest
from pathlib import Path
from healing_swarm.secrets import read_credentials

class CredentialTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text(text)
            return read_credentials(path)
    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(read_credentials(Path(directory)/'.env'), {})
    def test_plain_and_quoted_values(self):
        self.assertEqual(self.parse('# comment\nOPENROUTER_API_KEY=example # note\nexport RUNPOD_API_KEY="other"\n'), {'OPENROUTER_API_KEY':'example','RUNPOD_API_KEY':'other'})
    def test_values_are_not_expanded_or_executed(self):
        self.assertEqual(self.parse("OPENROUTER_API_KEY='$(whoami)$HOME'\n")['OPENROUTER_API_KEY'], '$(whoami)$HOME')
    def test_invalid_error_does_not_echo_secret(self):
        for text in ('WRONG=private-value', 'OPENROUTER_API_KEY="private-value'):
            with self.assertRaises(ValueError) as error:
                self.parse(text)
            self.assertNotIn('private-value', str(error.exception))
