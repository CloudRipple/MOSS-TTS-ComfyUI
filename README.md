# MOSS-TTS v1.5 ComfyUI 节点包

[English](README_en.md)

支持 MOSS-TTS v1.5 两个变体的 ComfyUI 自定义节点：

- **Local-Transformer**：Qwen3-**4B** 级主干 + nano-GPT2 局部变换器，配 MOSS-Audio-Tokenizer-v2，**48 kHz 立体声**，n_vq=12。（其他扩展 README 里流传的 "1.7B" 不准确——主干是 Qwen3-4B 形状，bf16 权重约 9.1GB。）
- **Delay**：8B delay-pattern 模型，配 MOSS-Audio-Tokenizer（v1），**24 kHz**，n_vq=32。

两个变体走同一组节点，Load Model 节点里下拉切换。

## 特性

- 无参考 TTS、零样本声音克隆、音频续写、硬时长控制（`target_tokens`，12.5 帧/秒）、31 种语言显式标签、`[pause 3.2s]` 停顿标记。
- **不用 `trust_remote_code`**：模型代码 vendored 在 `assets/`（最小且带标记的兼容补丁），不依赖 HF 模块缓存里那一版。
- transformers 4.x / 5.x 都能跑（补丁 + 特性探测）；双机型 E2E 验收覆盖 5.16 与 4.57。
- 深度接入 ComfyUI 内存管理：权重经 `ModelPatcher` / `ModelPatcherDynamic`（感知 AIMDO DynamicVRAM）注册，可被正常卸载；每个集成点都做了降级保护。
- 权重查找顺序：`$MOSS_TTS_MODELS_DIR/<仓库名>` → `ComfyUI/models/mosstts/<仓库名>` → HF hub cache（遵循 `HF_HOME`）→（`download_if_missing` 开启时）自动下载。

## 安装

**ComfyUI-Manager**：打开 Manager → Custom Nodes Manager → 搜索 `moss-tts` → Install。Manager 会自动拉包并安装依赖（`install.py`），装完重启 ComfyUI 即可。后续版本更新同样在 Manager 里完成。

**Git clone**：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CloudRipple/MOSS-TTS-ComfyUI.git
python install.py   # 只装缺失的轻量依赖
```

重启 ComfyUI。首次使用会先从本地缓存解析权重；找不到且允许下载时才从 HF 下载。

## 节点（分类 `MOSS-TTS v1.5`）

| 节点 | 作用 |
|---|---|
| Load Model | 选变体（local / delay）、dtype、attention、是否允许下载。 |
| Generate Speech | 无参考 TTS，语言 + instruction 引导声音。 |
| Voice Clone | 参考音频声音克隆。 |
| Continue Speech | 前缀续写；输出新段+拼好的完整音频+帧数（便于链式衔接）。 |
| Estimate Tokens | 文本 → `target_tokens` 估算。 |

生成节点都输出 `tokens_generated`（音频帧数，秒数 = 帧 / 12.5），供续写链精确传递前缀长度。

## 远程 API 节点（分类 `MOSS-TTS v1.5/Remote`）

不想本地加载模型时，也可以调用 Moss 平台（mosi.cn）托管 API：

| 节点 | 作用 |
|---|---|
| Remote Connect | 连接配置：base_url / API key（可留空走环境变量 `MOSS_API_KEY`）/ 模型（moss-tts-1.5-flash、moss-tts-1.0-pro）/ 异步开关 |
| Remote Speech | 远程单人 TTS，需填 voice_id；1.5-flash 支持文本内 `[pause 1.5s]` 停顿 |
| Remote Voice Clone | 参考音频在平台上克隆音色，输出 voice_id 给 Speech 用 |
| Remote Voice List | 列出账号已有音色（名称 + voice id） |

远程路径不支持：续写（Continue）、token 时长控制、采样参数——这些只有本地模型能做。

API key 在 [Moss 平台](https://platform.mosi.cn)控制台「API 密钥」页生成；现成音色可去 [Mossland 音色库](https://mossland.studio/voice/library) 挑，卡片上复制 voice id 填进 Remote Speech 即可。

## 显存

- Local（4B）：bf16 约 12 GB。
- Delay（8B）：bf16 约 22 GB。

克隆参考音频建议 5–15 秒；前缀越长，KV cache 越大。

## 同步上游代码

`assets/` 镜像了四个 HF 仓库的 remote code。所有补丁都带
`# MOSS-TTS-V15-ComfyUI patch:` 注释；同步上游时重新拷贝那四个仓库的代码文件并重新套用这组补丁即可。

## License

MIT（本包）。模型权重与上游代码为 OpenMOSS-Team 的 Apache-2.0。
