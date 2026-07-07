# Unified Model/Provider Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate provider+model configuration into a single Settings page, remove per-tab provider dropdowns, and fix audio provider bugs.

**Architecture:** Extend `gui/settings.py` with per-stage provider+model rows. Remove provider UI from stage tabs. Fix `call_audio_api()` to accept dynamic provider. Fix `get_available_providers()` audio filtering.

**Tech Stack:** Python, tkinter/ttkbootstrap, TOML config

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `config/default.toml` | Modify | Add `providers.stage_providers` defaults |
| `src/title_classifier/providers/__init__.py` | Modify | Fix `get_available_providers("audio")` filter; fix `call_audio_api()` to accept provider param |
| `src/title_classifier/gui/settings.py` | Refactor | Rewrite API section: per-stage provider+model rows, remove single provider/api_key |
| `src/title_classifier/gui/stage_refine.py` | Simplify | Remove `s1b_provider_var` and provider dropdown UI |
| `src/title_classifier/gui/stage_vision.py` | Simplify | Remove `s1c_provider_var` and provider dropdown UI |
| `src/title_classifier/gui/stage_audio.py` | Simplify | Remove `s1ca_provider_var` and provider dropdown UI |
| `src/title_classifier/core/refiner.py` | Adapt | Read provider from config instead of GUI variable |

---

### Task 1: Add `stage_providers` to config

**Files:**
- Modify: `config/default.toml`

- [ ] **Step 1: Add stage_providers section to default.toml**

Add the following after the existing `[providers]` section (after line 87):

```toml
[providers.stage_providers]
refine = "gcli"
vision = "gcli"
audio = "mimo"
```

- [ ] **Step 2: Verify TOML is valid**

Run: `python -c "import tomllib; tomllib.load(open('config/default.toml','rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add config/default.toml
git commit -m "config: add providers.stage_providers defaults"
```

---

### Task 2: Fix `get_available_providers()` audio filtering

**Files:**
- Modify: `src/title_classifier/providers/__init__.py:160-194`

- [ ] **Step 1: Add audio stage filter**

In `get_available_providers()`, add the audio filter after the existing `stage == "1c"` check at line 178:

```python
if stage == "audio" and not config.get("supports_audio", False):
    continue
```

The function should now handle all three stages: `"1b"`, `"1c"`, `"audio"`.

- [ ] **Step 2: Verify import works**

Run: `python -c "from title_classifier.providers import get_available_providers; print(len(get_available_providers('audio')), 'audio providers')"`
Expected: Only providers with `supports_audio=True` and valid API keys appear (likely 1: mimo)

- [ ] **Step 3: Commit**

```bash
git add src/title_classifier/providers/__init__.py
git commit -m "fix: filter providers by supports_audio in get_available_providers()"
```

---

### Task 3: Fix `call_audio_api()` to accept dynamic provider

**Files:**
- Modify: `src/title_classifier/providers/__init__.py:664-693`

- [ ] **Step 1: Add provider_name parameter to call_audio_api()**

Change the function signature (line 664) to:

```python
def call_audio_api(
    audio_b64: str,
    prompt: str = None,
    model: str = None,
    api_key: str = None,
    provider_name: str = "mimo",
    timeout: int = 120,
    retries: int = 3,
    reasoning_retries: int = 2,
) -> str:
```

Replace the hardcoded values at lines 688-693:

```python
# Before (hardcoded):
api_key = api_key or get_api_key("mimo")
...
api_url = "https://api.xiaomimimo.com/v1/chat/completions"

# After (dynamic):
api_key = api_key or get_api_key(provider_name)
if not api_key:
    return f"[错误] 缺少 {provider_name.upper()}_API_KEY"

provider_config = get_provider_config(provider_name) or {}
model = model or _resolve_stage_model(provider_config, "audio") or "mimo-v2.5"
api_url = f"{provider_config.get('url', '')}/v1/chat/completions"
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from title_classifier.providers import call_audio_api; print('import OK')"`
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add src/title_classifier/providers/__init__.py
git commit -m "fix: call_audio_api accepts provider_name instead of hardcoding mimo"
```

---

### Task 4: Update `_resolve_stage_model()` to also read stage_providers

**Files:**
- Modify: `src/title_classifier/providers/__init__.py:319-330`

- [ ] **Step 1: Update _resolve_stage_model to read stage_providers**

No change needed to `_resolve_stage_model` itself — it already reads `providers.models.<stage>` from config. The stage_providers config is read by the Settings UI and passed down. This task is a no-op verification step.

Verify that `_resolve_stage_model` correctly reads from config:

```python
def _resolve_stage_model(provider_config: dict, stage: str = None) -> str:
    """获取阶段特定模型，fallback 到 provider 默认模型"""
    if stage:
        try:
            from ..utils.config import load_merged_config
            cfg = load_merged_config()
            stage_model = cfg.get("providers", {}).get("models", {}).get(stage, "")
            if stage_model:
                return stage_model
        except Exception:
            pass
    return provider_config.get("default_model", "")
```

This already works correctly. No changes needed.

- [ ] **Step 2: Commit**

No commit needed — this is a verification step only.

---

### Task 5: Refactor Settings UI — per-stage provider+model rows

**Files:**
- Modify: `src/title_classifier/gui/settings.py`

- [ ] **Step 1: Replace _init_vars() variables**

Replace lines 55-63 with:

```python
        # 分阶段 Provider + Model 配置
        self.stage_refine_provider_var = tk.StringVar()
        self.stage_refine_model_var = tk.StringVar()
        self.stage_vision_provider_var = tk.StringVar()
        self.stage_vision_model_var = tk.StringVar()
        self.stage_audio_provider_var = tk.StringVar()
        self.stage_audio_model_var = tk.StringVar()
        self.timeout_var = tk.StringVar()
```

Remove `self.provider_var`, `self.api_key_var`, `self.model_refine_var`, `self.model_vision_var`, `self.model_audio_var`.

- [ ] **Step 2: Rewrite _build_api_section()**

Replace the entire `_build_api_section` method (lines 148-228) with:

```python
    def _build_api_section(self, parent):
        """API 配置区块 — 每阶段独立 Provider + Model"""
        frame = ttk.LabelFrame(parent, text="模型配置")
        frame.pack(fill=tk.X, pady=(0, 10), padx=10)

        stages = [
            ("标题优化 (Refine)", "refine", self.stage_refine_provider_var, self.stage_refine_model_var),
            ("视觉识别 (Vision)", "vision", self.stage_vision_provider_var, self.stage_vision_model_var),
            ("音频识别 (Audio)", "audio", self.stage_audio_provider_var, self.stage_audio_model_var),
        ]

        self._stage_model_combos = {}

        for label, stage_key, provider_var, model_var in stages:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=4)

            ttk.Label(row, text=f"{label}:", width=16).pack(side=tk.LEFT)

            # Provider dropdown
            try:
                from ..providers import get_providers_for_gui
                providers = get_providers_for_gui(stage_key)
            except Exception:
                providers = ["gcli", "mimo"]
            provider_combo = ttk.Combobox(row, textvariable=provider_var, values=providers, state="readonly", width=12)
            provider_combo.pack(side=tk.LEFT, padx=4)

            # Model dropdown
            model_combo = ttk.Combobox(row, textvariable=model_var, width=28)
            model_combo.pack(side=tk.LEFT, padx=4)
            self._stage_model_combos[stage_key] = model_combo

            # Fetch button
            ttk.Button(row, text="🔄", width=3,
                       command=lambda s=stage_key, pv=provider_var: self._fetch_models_for_stage(s, pv)).pack(side=tk.LEFT, padx=2)

            # Update model list when provider changes
            provider_combo.bind("<<ComboboxSelected>>",
                                lambda e, s=stage_key, pv=provider_var: self._on_provider_changed(s, pv))

        # 超时
        row_timeout = ttk.Frame(frame)
        row_timeout.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(row_timeout, text="API 超时(秒):", width=16).pack(side=tk.LEFT)
        ttk.Entry(row_timeout, textvariable=self.timeout_var, width=6).pack(side=tk.LEFT, padx=4)

        # 高级配置按钮
        row_adv = ttk.Frame(frame)
        row_adv.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(row_adv, text="高级 API 配置...", command=self._open_api_config, bootstyle=INFO).pack(side=tk.LEFT)
        ttk.Label(row_adv, text="配置各 Provider 的 URL、API Key", foreground="#888888", font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=10)
```

- [ ] **Step 3: Add helper methods for provider/model interaction**

Add these methods to the class:

```python
    def _on_provider_changed(self, stage_key, provider_var):
        """切换 Provider 时清空 Model"""
        self._stage_model_combos[stage_key].set("")

    def _fetch_models_for_stage(self, stage_key, provider_var):
        """获取指定阶段 Provider 的模型列表"""
        import threading
        provider = provider_var.get()
        if not provider:
            messagebox.showwarning("提示", "请先选择 Provider", parent=self)
            return

        combo = self._stage_model_combos[stage_key]

        def do_fetch():
            try:
                from ..providers import get_provider_config, get_api_key
                import json, ssl, http.client
                from urllib.parse import urlparse

                config = get_provider_config(provider)
                api_key = get_api_key(provider)
                api_url = config.get("url", "")
                if not api_url:
                    raise Exception("URL 为空")

                parsed = urlparse(api_url)
                models_url = f"{parsed.scheme}://{parsed.hostname}/v1/models"

                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                conn = http.client.HTTPSConnection(parsed.hostname, context=ctx, timeout=15)
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = "Bearer " + api_key

                conn.request("GET", "/v1/models", headers=headers)
                resp = conn.getresponse()
                data = resp.read().decode()
                conn.close()

                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")

                result = json.loads(data)
                model_list = [m.get("id", "") for m in result.get("data", [])]

                def update_ui():
                    if model_list:
                        combo["values"] = model_list
                    else:
                        messagebox.showinfo("提示", "无可用模型", parent=self)
                self.after(0, update_ui)

            except Exception as e:
                def show_error():
                    messagebox.showerror("错误", f"获取模型失败: {e}", parent=self)
                self.after(0, show_error)

        threading.Thread(target=do_fetch, daemon=True).start()
```

- [ ] **Step 4: Update _load_values() to read stage_providers**

Replace lines 302-310 with:

```python
        # 分阶段 Provider + Model
        self.stage_refine_provider_var.set(get_config_value(c, "providers.stage_providers.refine", "gcli"))
        self.stage_refine_model_var.set(get_config_value(c, "providers.models.refine", ""))
        self.stage_vision_provider_var.set(get_config_value(c, "providers.stage_providers.vision", "gcli"))
        self.stage_vision_model_var.set(get_config_value(c, "providers.models.vision", ""))
        self.stage_audio_provider_var.set(get_config_value(c, "providers.stage_providers.audio", "mimo"))
        self.stage_audio_model_var.set(get_config_value(c, "providers.models.audio", ""))
        self.timeout_var.set(str(get_config_value(c, "providers.timeout", 90)))
```

Remove the old `self.provider_var.set(...)` and `self.api_key_var.set(...)` lines.

- [ ] **Step 5: Update _save() to write stage_providers**

Replace the providers section in `_save()` (lines 333-341) with:

```python
            "providers": {
                "timeout": int(self.timeout_var.get() or 90),
                "stage_providers": {
                    "refine": self.stage_refine_provider_var.get(),
                    "vision": self.stage_vision_provider_var.get(),
                    "audio": self.stage_audio_provider_var.get(),
                },
                "models": {
                    "refine": self.stage_refine_model_var.get(),
                    "vision": self.stage_vision_model_var.get(),
                    "audio": self.stage_audio_model_var.get(),
                },
            },
```

Remove the `"api": {"key": ...}` section from the save dict.

- [ ] **Step 6: Remove _fetch_available_models() and _toggle_api_key_visibility()**

Delete the old `_fetch_available_models()` method (lines 399-465) and `_toggle_api_key_visibility()` method (lines 467-472). These are replaced by `_fetch_models_for_stage()`.

- [ ] **Step 7: Verify UI loads without errors**

Run: `python -c "import tkinter; root = tkinter.Tk(); root.withdraw(); from title_classifier.gui.settings import SettingsDialog; print('import OK')"`
Expected: `import OK`

- [ ] **Step 8: Commit**

```bash
git add src/title_classifier/gui/settings.py
git commit -m "refactor: settings page with per-stage provider+model rows"
```

---

### Task 6: Remove provider dropdown from Stage Refine tab

**Files:**
- Modify: `src/title_classifier/gui/stage_refine.py`

- [ ] **Step 1: Remove provider variable and dropdown**

Remove `self.s1b_provider_var = tk.StringVar(value="gcli")` (line 30).

Remove the import of `get_providers_for_gui` (line 11).

Remove these lines from `_build_ui()` (lines 59-61):

```python
        ttk.Label(file_bar, text="AI:").pack(side=tk.LEFT, padx=(0, 2))
        providers = get_providers_for_gui("1b")
        ttk.Combobox(file_bar, textvariable=self.s1b_provider_var, values=providers, state="readonly", width=8).pack(side=tk.LEFT, padx=2)
```

- [ ] **Step 2: Update _run_refine() to read provider from config**

In `_run_refine()` (line 383), replace:

```python
        provider = self.s1b_provider_var.get()
```

with:

```python
        from ..utils.config import load_merged_config, get_config_value
        config = load_merged_config()
        provider = get_config_value(config, "providers.stage_providers.refine", "gcli")
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from title_classifier.gui.stage_refine import StageRefineTab; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/title_classifier/gui/stage_refine.py
git commit -m "refactor: remove provider dropdown from stage_refine tab"
```

---

### Task 7: Remove provider dropdown from Stage Vision tab

**Files:**
- Modify: `src/title_classifier/gui/stage_vision.py`

- [ ] **Step 1: Remove provider variable and dropdown**

Remove the import of `get_providers_for_gui` (line 16).

Remove these lines from `_build()` (lines 111-114):

```python
        ttk.Label(top_bar, text="AI:").pack(side=tk.LEFT, padx=(0, 2))
        self.s1c_provider_var = tk.StringVar(value="gcli")
        providers = get_providers_for_gui("1c")
        ttk.Combobox(top_bar, textvariable=self.s1c_provider_var, values=providers, state="readonly", width=8).pack(side=tk.LEFT, padx=2)
```

Also remove the separator at line 109:
```python
        ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
```

- [ ] **Step 2: Update _run_vision() to read provider from config**

In `_run_vision()` (line 247), replace:

```python
        provider = self.s1c_provider_var.get()
```

with:

```python
        from ..utils.config import load_merged_config, get_config_value
        config = load_merged_config()
        provider = get_config_value(config, "providers.stage_providers.vision", "gcli")
```

Apply the same change in `_run_vision_retry()` (line 335).

- [ ] **Step 3: Verify import works**

Run: `python -c "from title_classifier.gui.stage_vision import StageVisionTab; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/title_classifier/gui/stage_vision.py
git commit -m "refactor: remove provider dropdown from stage_vision tab"
```

---

### Task 8: Remove provider dropdown from Stage Audio tab

**Files:**
- Modify: `src/title_classifier/gui/stage_audio.py`

- [ ] **Step 1: Remove provider variable and dropdown**

Remove the import of `get_providers_for_gui` (line 57).

Remove the entire Provider selection frame (lines 52-61):

```python
        # Provider选择
        provider_frame = ttk.LabelFrame(tab, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1ca_provider_var = tk.StringVar(value="mimo")
        from ..providers import get_providers_for_gui
        providers = get_providers_for_gui("audio")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.s1ca_provider_var, values=providers, state="readonly")
        provider_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(provider_combo, "选择音频AI服务提供商\n- mimo: 小米MiMo（推荐，支持音频理解）")
```

- [ ] **Step 2: Update audio processing to read provider from config**

In the audio task runner (around line 201), replace:

```python
        provider = self.s1ca_provider_var.get()
```

with:

```python
        from ..utils.config import load_merged_config, get_config_value
        config_cfg = load_merged_config()
        provider = get_config_value(config_cfg, "providers.stage_providers.audio", "mimo")
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from title_classifier.gui.stage_audio import StageAudioTab; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/title_classifier/gui/stage_audio.py
git commit -m "refactor: remove provider dropdown from stage_audio tab"
```

---

### Task 9: Verify Refiner reads provider correctly

**Files:**
- Verify: `src/title_classifier/core/refiner.py`

- [ ] **Step 1: Verify Refiner receives provider from caller**

The `Refiner` class (line 22) takes `provider` as a constructor arg: `def __init__(self, provider: str = "gcli")`. After Task 6, the caller in `stage_refine.py` will read from config and pass it. No changes needed to `refiner.py` itself.

Verify: `grep -n "Refiner(provider" src/title_classifier/gui/stage_refine.py`
Expected: Shows the updated line that reads from config.

- [ ] **Step 2: Commit**

No commit needed — verification only.

---

### Task 10: End-to-end verification

- [ ] **Step 1: Run the application and verify Settings page**

Run: `python -m title_classifier` (or the project's entry point)

Verify:
1. Settings page shows 3 rows: Refine, Vision, Audio — each with Provider + Model + Fetch button
2. Provider dropdowns show correct filtered providers (e.g., audio only shows providers with `supports_audio`)
3. Fetch button populates model list for that specific stage
4. Save writes correct config to `user.toml`

- [ ] **Step 2: Verify stage tabs have no provider dropdown**

Verify:
1. Stage Refine tab — no "AI:" dropdown
2. Stage Vision tab — no "AI:" dropdown
3. Stage Audio tab — no "AI Provider" section

- [ ] **Step 3: Verify audio processing uses configured provider**

Run audio recognition and verify it uses the provider from Settings (not hardcoded mimo).

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: unified model/provider configuration (closes #6)"
```
