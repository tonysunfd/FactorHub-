from __future__ import annotations

import os
from pathlib import Path

from backend.core.settings import settings


class LLMConfigService:
    def _env_path(self) -> Path:
        return settings.BASE_DIR / '.env'

    def _read_env_values(self) -> dict[str, str]:
        env_path = self._env_path()
        values: dict[str, str] = {}
        if not env_path.exists():
            return values
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
        return values

    def _write_env_values(self, updates: dict[str, str]) -> None:
        env_path = self._env_path()
        lines: list[str] = []
        if env_path.exists():
            lines = env_path.read_text(encoding='utf-8').splitlines()

        seen: set[str] = set()
        new_lines: list[str] = []
        for line in lines:
            replaced = False
            for key, value in updates.items():
                if line.startswith(f'{key}='):
                    new_lines.append(f'{key}={value}')
                    seen.add(key)
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)

        for key, value in updates.items():
            if key not in seen:
                new_lines.append(f'{key}={value}')

        env_path.write_text('\n'.join(new_lines).rstrip() + '\n', encoding='utf-8')

    async def get_config(self) -> dict:
        env_values = self._read_env_values()
        api_key = (env_values.get('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_API_KEY') or '').strip()
        base_url = (env_values.get('DEEPSEEK_BASE_URL') or os.getenv('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com/v1').strip()
        model = (env_values.get('DEEPSEEK_MODEL') or os.getenv('DEEPSEEK_MODEL') or 'deepseek-chat').strip()
        return {
            'success': True,
            'base_url': base_url,
            'model': model,
            'execution_mode': 'embedded_factorhub_backend',
            'has_api_key': bool(api_key),
            'api_key_masked': f"{api_key[:4]}***{api_key[-4:]}" if len(api_key) >= 8 else ('*' * len(api_key) if api_key else ''),
            'message': '当前配置直接供 FactorHub 本地后端内置 LLM 使用；保存后，新任务会自动读取最新配置。',
        }

    async def update_config(self, payload: dict) -> dict:
        base_url = (payload.get('base_url') or 'https://api.deepseek.com/v1').strip()
        model = (payload.get('model') or 'deepseek-chat').strip()
        api_key = payload.get('api_key')

        updates = {
            'DEEPSEEK_BASE_URL': base_url,
            'DEEPSEEK_MODEL': model,
        }
        if api_key is not None and api_key.strip():
            updates['DEEPSEEK_API_KEY'] = api_key.strip()

        self._write_env_values(updates)
        for key, value in updates.items():
            os.environ[key] = value

        result = await self.get_config()
        result['message'] = 'LLM 配置已保存，后续在 FactorHub 本地后端发起的新自动挖掘/自动回测任务会直接使用该配置。'
        return result

    async def restart_quantgpt(self) -> dict:
        return {
            'success': True,
            'embedded': True,
            'message': '当前已切换为 FactorHub 本地内置 QuantGPT/LLM 模式，无需重启外部 QuantGPT 进程。',
        }


llm_config_service = LLMConfigService()
