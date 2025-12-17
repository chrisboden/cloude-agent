import os
import secrets
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: save_transcript.py <transcript_text>")

    transcript_text = sys.argv[1]

    workspace_dir = Path.cwd()
    transcripts_dir = workspace_dir / "artifacts" / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%d-%H%M%S", time.gmtime())
    suffix = secrets.token_hex(4)
    filename = f"{ts}-{suffix}-transcript.txt"
    path = transcripts_dir / filename

    path.write_text(transcript_text)

    base_url = os.environ.get("PUBLIC_BASE_URL").rstrip("/")
    print(f"{base_url}/artifacts/transcripts/{filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
