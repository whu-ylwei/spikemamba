`model_best.pth.tar` is split into:

- `model_best.pth.tar.part-00`
- `model_best.pth.tar.part-01`
- `model_best.pth.tar.part-02`

Rebuild command:

```bash
cat model_best.pth.tar.part-00 model_best.pth.tar.part-01 model_best.pth.tar.part-02 > model_best.pth.tar
```

SHA256 of rebuilt file:

```text
4388e2224d29c94cb10d593c426cf5434fe1f285fe6d11a1780ec4b4742ec246
```
