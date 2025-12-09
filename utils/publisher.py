#!/usr/bin/env python3
"""
Results publisher
Upload trained and packaged paper materials to a backend (e.g., results.auto.ml)
"""

from pathlib import Path
from typing import Dict, Optional
import os
import json
import shutil
import tempfile
import zipfile

try:
    import requests
except Exception:
    requests = None


class ResultsPublisher:
    """Publisher that uploads results to a remote backend"""

    def __init__(self,
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.api_base = api_base or os.getenv('RESULTS_API_URL', '').strip()
        self.api_key = api_key or os.getenv('RESULTS_API_KEY', '').strip()

    def is_configured(self) -> bool:
        return bool(self.api_base) and requests is not None

    def _zip_dir(self, directory: Path) -> Path:
        directory = Path(directory)
        tmp_dir = Path(tempfile.mkdtemp(prefix='results_upload_'))
        zip_path = tmp_dir / f"{directory.name}.zip"
        exclude_dirs = {"docs"}
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for root, dirnames, filenames in os.walk(directory):
                dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
                for fname in filenames:
                    fpath = Path(root) / fname
                    arcname = fpath.relative_to(directory)
                    zf.write(fpath, arcname)
        return zip_path

    def publish_package(self,
                        package_dir: str,
                        metadata: Optional[Dict] = None,
                        endpoint: str = '/api/v1/results/upload') -> Optional[Dict]:
        """
        Upload zipped package with metadata
        Returns response JSON or None
        """
        if not self.is_configured():
            print("WARNING: ResultsPublisher not configured or requests missing; skipping publish")
            return None

        package_dir = Path(package_dir)
        if not package_dir.exists():
            print(f"ERROR: Package directory not found: {package_dir}")
            return None

        try:
            zip_path = self._zip_dir(package_dir)
        except Exception as e:
            print(f"ERROR: Packaging failed: {e}")
            return None

        url = self.api_base.rstrip('/') + endpoint
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        files = {
            'package': (zip_path.name, open(zip_path, 'rb'), 'application/zip')
        }
        data = {
            'metadata': json.dumps(metadata or {}, ensure_ascii=True)
        }

        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
            if resp.status_code >= 200 and resp.status_code < 300:
                try:
                    return resp.json()
                except Exception:
                    return {'status': 'ok', 'text': resp.text}
            else:
                print(f"ERROR: Publish failed: HTTP {resp.status_code} - {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"ERROR: Publish exception: {e}")
            return None
