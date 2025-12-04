#!/usr/bin/env python3
"""
结果发布器
将训练与打包好的论文资料包发布到后端（如 results.auto.ml）
"""

from pathlib import Path
from typing import Dict, Optional
import os
import json
import shutil
import tempfile

try:
    import requests
except Exception:
    requests = None


class ResultsPublisher:
    """将结果上传到远端的发布器"""

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
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', root_dir=str(directory))
        return zip_path

    def publish_package(self,
                        package_dir: str,
                        metadata: Optional[Dict] = None,
                        endpoint: str = '/api/v1/results/upload') -> Optional[Dict]:
        """
        上传整个资料包（zip）与元数据
        Returns 响应JSON或None
        """
        if not self.is_configured():
            print("⚠️ ResultsPublisher 未配置或缺少 requests，跳过发布")
            return None

        package_dir = Path(package_dir)
        if not package_dir.exists():
            print(f"❌ 资料包目录不存在: {package_dir}")
            return None

        try:
            zip_path = self._zip_dir(package_dir)
        except Exception as e:
            print(f"❌ 打包失败: {e}")
            return None

        url = self.api_base.rstrip('/') + endpoint
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        files = {
            'package': (zip_path.name, open(zip_path, 'rb'), 'application/zip')
        }
        data = {
            'metadata': json.dumps(metadata or {}, ensure_ascii=False)
        }

        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
            if resp.status_code >= 200 and resp.status_code < 300:
                try:
                    return resp.json()
                except Exception:
                    return {'status': 'ok', 'text': resp.text}
            else:
                print(f"❌ 发布失败: HTTP {resp.status_code} - {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"❌ 发布异常: {e}")
            return None


