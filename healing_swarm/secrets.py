"""Read local credentials as data, without shell execution or environment export."""
from pathlib import Path

ALLOWED_KEYS = {'OPENROUTER_API_KEY', 'RUNPOD_API_KEY'}

def read_credentials(path):
    path = Path(path)
    if not path.exists():
        return {}
    credentials = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or key not in ALLOWED_KEYS:
            raise ValueError(f'Invalid credential entry on .env line {number}')
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f'Invalid quoted credential on .env line {number}')
            value = value[1:-1]
        else:
            value = value.split(' #', 1)[0].rstrip()
        # No interpolation, shell expansion, or logging of credential values.
        credentials[key] = value
    return credentials
