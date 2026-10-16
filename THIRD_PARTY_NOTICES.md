# Third-party code

The two source trees contain modified upstream projects. Existing copyright and attribution notices in individual files are retained.

| Directory | Upstream project | Upstream license |
| --- | --- | --- |
| `swin/` | [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer) | [MIT](swin/LICENSE) |
| `vit/` | [PyTorch Image Models by Ross Wightman and contributors](https://github.com/huggingface/pytorch-image-models) | [Apache-2.0](vit/LICENSE) |

The original project READMEs remain in their respective directories. The license files were restored from the upstream repositories because they were absent from the supplied source copies.

EQ additions include the `swin_transformer_eq_*` modules and supporting layers in `swin/models/`, and the `vision_transformer_eq*` modules in `vit/timm/models/`. The model builders, registration, and Swin configuration connect these additions to the upstream training code. The original author notices in `swin/models/e_linear.py` and `swin/models/B_Conv_woregis.py` are retained.

Repository preparation also adds dependency lists and usage documentation, corrects argument forwarding in `vit/distributed_train.sh`, and removes the duplicate EQ model import from `vit/timm/models/__init__.py` while retaining the previously effective implementation. These upstream notices do not assign a new blanket license to separately authored contributions.
