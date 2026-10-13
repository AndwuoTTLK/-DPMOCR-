在实习期间用Python编写的基于WeChat QR识别二维码，Paddle-OCRv6识别工业点阵字，OpenCV保底矫正的工业钢板代号识别项目。
项目目标图像由两部分组成：二维码和右侧的代码信息。
由于二维码直接指向右侧代码信息，所以优先执行二维码扫描，扫描成功后输出二维码内文字信息后就不会再进行文字识别，当二维码因为拍摄不清楚等原因无法正常识别时，会先使用基于PP-OCRv4 的模型检测文字行 → 合并 / 过滤 / 拆高框，再使用基于Paddle-OCRv6训练的模型来进行文字识别。目前例图识别成功率在90%左右。
图形界面调试模式下的分行图、预处理图等会生成在 debug_preprocess 中；终端批量识别默认不生成这些中间图。

主要用于练手和熟悉开发流程

测试效果用图在Test文件夹中，会在后续上传

上传1.1.0版本，增加了一点性能优化
上传1.1.1版本，增加打包修复和忙碌锁

## 终端批量识别

本地项目目录为 `D:\RQment\WORD\毕业论文\-DPMOCR-`。首次在另一台电脑克隆后，运行 `py -3.12 -m venv .venv` 和 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` 安装依赖；模型文件位于 `model` 文件夹。图形界面可运行 `.\run_ocr.cmd`，详细步骤见 `运行说明.txt`。

在项目目录打开 PowerShell，无需启动图形界面。可以一次传入多个图片、文件夹或通配符；文件夹会递归扫描，重复图片只处理一次：

```powershell
.\run_batch.cmd ".\Test" -o ".\batch_results"
.\run_batch.cmd ".\图片1.png" ".\图片2.jpg" -o ".\batch_results"
.\run_batch.cmd ".\待识别\*.png" -o ".\batch_results"
```

也可以直接运行 `.venv\Scripts\python.exe batch_recognize.py`，参数相同。支持 PNG、JPG、JPEG、BMP、TIF、TIFF 和 WEBP。

程序在终端逐张显示二维码或文字内容，并在输出文件夹生成两个 UTF-8 文件：

- `results.jsonl`：每行对应一张图片，包含原图绝对路径、处理状态、宽高、二维码数组、文字数组及错误信息。适合程序逐行读取。
- `results.csv`：每个识别项一行，列包含图片路径、类型、文本、置信度及位置；失败或未识别的图片也会保留一行。使用 UTF-8 BOM，便于 Excel 直接打开。

状态 `ok` 表示识别到内容，`no_result` 表示图片处理成功但未识别到内容，`error` 表示加载或识别出错。单张图片出错不会中断整个批次，但命令最终返回非零退出码。再次写入同一输出文件夹时，会生成 `results_002.*` 等新文件，不覆盖旧结果。
